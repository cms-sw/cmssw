// -*- C++ -*-
//
// Package:     MagneticField/GeomBuilder
// Class  :     InterpolatorBuilder
//
// Implementation:
//     [Notes on implementation]
//
// Original Author:  Christopher Jones
//         Created:  Tue, 17 May 2022 20:50:21 GMT
//

// system include files

// user include files
#include "InterpolatorBuilder.h"
#include "FakeInterpolator.h"

#include "FWCore/ParameterSet/interface/FileInPath.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "MagneticField/Interpolation/interface/binary_ifstream.h"
#include "MagneticField/Interpolation/interface/MFGridFactory.h"
#include "MagneticField/Interpolation/interface/MFGrid.h"

#include "DataFormats/Math/interface/angle_units.h"
//
// constants, enums and typedefs
//

namespace magneticfield {
  using namespace angle_units::operators;

  //
  // static data member definitions
  //

  //
  // constructors and destructor
  //
  InterpolatorBuilder::InterpolatorBuilder(std::string iTableSet, bool useMergeFileIfAvailable)
      : tableSet_(std::move(iTableSet)) {
    if (not useMergeFileIfAvailable)
      return;
    auto indexFileName = edm::FileInPath::findFile("MagneticField/Interpolation/data/" + tableSet_ + "/merged.index");
    if (not indexFileName.empty()) {
      auto binaryFileName = edm::FileInPath::findFile("MagneticField/Interpolation/data/" + tableSet_ + "/merged.bin");
      if (not binaryFileName.empty()) {
        std::ifstream indexFile(indexFileName);
        while (indexFile) {
          std::string magFile;
          unsigned int offset;
          indexFile >> magFile >> offset;
          offsets_.emplace(std::move(magFile), offset);
        }
        stream_ = interpolation::binary_ifstream(binaryFileName);
      }
    }
  }

  //
  // member functions
  //
  std::unique_ptr<MagProviderInterpol> InterpolatorBuilder::build(volumeHandle const* vol) {
    if (tableSet_ == "fake" || vol->magFile == "fake") {
      return std::make_unique<magneticfield::FakeInterpolator>();
    }

    // If the table is in "local" coordinates, must create a reference
    // frame that is appropriately rotated along the CMS Z axis.

    GloballyPositioned<float> rf = *(vol->placement());

    if (vol->masterSector != 1) {
      typedef Basic3DVector<float> Vector;

      // Phi of the master sector
      double masterSectorPhi = (vol->masterSector - 1) * 1._pi / 6.;

      GloballyPositioned<float>::RotationType rot(Vector(0, 0, 1), -masterSectorPhi);
      Vector vpos(vol->placement()->position());

      rf = GloballyPositioned<float>(GloballyPositioned<float>::PositionType(rot.multiplyInverse(vpos)),
                                     vol->placement()->rotation() * rot);
    }

    if (not stream_) {
      auto fullPath = edm::FileInPath::findFile("MagneticField/Interpolation/data/" + tableSet_ + "/" + vol->magFile);
      if (fullPath.empty()) {
        //cause the exception to happen
        edm::FileInPath mydata("MagneticField/Interpolation/data/" + tableSet_ + "/" + vol->magFile);
        return {};
      }

      magneticfield::interpolation::binary_ifstream strm(fullPath);
      return std::unique_ptr<MagProviderInterpol>(MFGridFactory::build(strm, rf));
    }

    auto find = offsets_.find(vol->magFile);
    if (find == offsets_.end()) {
      throw cms::Exception("MissingMagFileEntry") << vol->magFile << " was not an entry in the index file";
    }
    stream_->seekg(find->second);
    if (stream_->fail()) {
      throw cms::Exception("SeekMagFileEntry") << " failed seekg within merged binary file";
    }
    return std::unique_ptr<MagProviderInterpol>(MFGridFactory::build(*stream_, rf));
  }

  //
  // const member functions
  //

  //
  // static member functions
  //
}  // namespace magneticfield
