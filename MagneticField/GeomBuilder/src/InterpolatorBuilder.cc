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
  InterpolatorBuilder::InterpolatorBuilder(std::string iTableSet) : tableSet_(std::move(iTableSet)) {}

  //
  // member functions
  //
  std::unique_ptr<MagProviderInterpol> InterpolatorBuilder::build(volumeHandle const* vol) {
    if (tableSet_ == "fake" || vol->magFile == "fake") {
      return std::make_unique<magneticfield::FakeInterpolator>();
    }

    auto fullPath = edm::FileInPath::findFile("MagneticField/Interpolation/data/" + tableSet_ + "/" + vol->magFile);
    if (fullPath.empty()) {
      //cause the exception to happen
      edm::FileInPath mydata("MagneticField/Interpolation/data/" + tableSet_ + "/" + vol->magFile);
      return {};
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

    magneticfield::interpolation::binary_ifstream strm(fullPath);
    return std::unique_ptr<MagProviderInterpol>(MFGridFactory::build(strm, rf));
  }

  //
  // const member functions
  //

  //
  // static member functions
  //
}  // namespace magneticfield
