/*
 *  See header file for a description of this class.
 *
 *  \author N. Amapane - INFN Torino
 */

#include "bLayer.h"
#include "printUniqueNames.h"
#include "MagneticField/VolumeGeometry/interface/MagVolume6Faces.h"
#include "MagneticField/Layers/interface/MagBLayer.h"
#include "MagneticField/Layers/interface/MagVerbosity.h"

#include "Utilities/General/interface/precomputed_value_sort.h"

#include <iostream>

using namespace SurfaceOrientation;
using namespace magneticfield;

//The ctor is in charge of finding sectors inside the layer.
bLayer::bLayer(handles::const_iterator begin, handles::const_iterator end, bool debugFlag)
    : size(end - begin), theVolumes(begin, end), mlayer(nullptr), debug(debugFlag) {
  // Sort in phi
  precomputed_value_sort(theVolumes.begin(), theVolumes.end(), ExtractPhi());

  if (debug) {
    std::cout << " elements: " << theVolumes.size() << " unique volumes: ";
    printUniqueNames(theVolumes.begin(), theVolumes.end());
  }

  // Find sectors in phi
  handles::iterator secBegin = theVolumes.begin();
  handles::iterator secEnd = secBegin;
  int binOffset = 0;

  const Surface& refSurf = (*secBegin)->surface(outer);

  int newbegin = 0;
  int newend = 0;

  // A sector is made of several volumes in R, and, for planar layers
  // (box and traps) also in phi, so it might cross the -phi boundary.
  // For those, we have to look for the end of first sector and rotate the
  // vector of volumes.
  // ASSUMPTION: all volumes in a layer must be of compatible type.

  if (size == 1) {  // Only one volume; this is the case for barrel
    // cylinders.
    // FIXME sectors.push_back(bSector(theVolumes.begin(),theVolumes.end());
    if (debug)
      std::cout << "      Sector is just one volume." << std::endl;

  } else if (size == 12 ||  // In this case, each volume is a sector.
             (((*secBegin)->shape() != DDSolidShape::ddtrap) && (*secBegin)->shape() != DDSolidShape::ddbox)) {
    secEnd = secBegin + size / 12;

  } else {                    // there are more than one volume per sector.
    float tolerance = 0.025;  // 250 micron
    do {
      if (debug)
        std::cout << (*secBegin)->name << " " << (*secBegin)->copyno << std::endl;
      ++secBegin;
    } while (
        (secBegin != theVolumes.end()) &&
        (*secBegin)->sameSurface(
            refSurf,
            outer,
            tolerance));  // This works only if outer surface is a plane, otherwise sameSurface returns always true!

    secEnd = secBegin;
    secBegin = theVolumes.begin() + bin((secEnd - theVolumes.begin()) - size / 12);
    ;
    newend = secEnd - theVolumes.begin();
    newbegin = secBegin - theVolumes.begin();

    // Rotate the begin of the first sector to the vector beginning.
    rotate(theVolumes.begin(), secBegin, theVolumes.end());
    secBegin = theVolumes.begin();
    secEnd = secBegin + size / 12;

    // Test it is correct...
    if (!((*secBegin)->sameSurface((*(secEnd - 1))->surface(outer), outer, tolerance))) {
      std::cout << "*** ERROR: Big mess while looking for sectors " << (*secBegin)->name << " " << (*secBegin)->copyno
                << " " << (*(secEnd - 1))->name << " " << (*(secEnd - 1))->copyno << std::endl;
    }
  }

  if (debug) {
    std::cout << "      First sector: volumes " << secEnd - theVolumes.begin() << " from " << newbegin
              << " (phi = " << (*secBegin)->center().phi() << ") "
              << " to " << newend << " (phi = " << (*secEnd)->center().phi() << ") "
              << " # " << (*secBegin)->copyno << " ";
    std::cout << GlobalVector(refSurf.rotation().zx(), refSurf.rotation().zy(), refSurf.rotation().zz()) << std::endl;
  }

  if (size != 1) {  // Build the 12 sectors
    int offset = size / 12;
    sectors.resize(12);
    for (int i = 0; i < 12; ++i) {
      int isec = (i + binOffset) % 12;
      sectors[isec >= 0 ? isec : isec + 12] =
          bSector(theVolumes.begin() + ((i)*offset), theVolumes.begin() + ((i + 1) * offset), debug);
    }
  }

  if (debug)
    std::cout << "-----------------------" << std::endl;
}

int bLayer::bin(int i) const {
  i = i % size;
  return (i >= 0 ? i : i + size);
}

// const bSector &
// bLayer::sector(int i) const {
//   i = i%12;
//   return sectors[i>=0?i:i+12];
// }

double bLayer::minR() const {
  // ASSUMPTION: a layer is only 1 volume thick (by construction).
  return theVolumes.front()->minR();
}

// double bLayer::maxR() const {
//   // ASSUMPTION: a layer is only 1 volume thick (by construction).
//   return theVolumes.front()->maxR();
// }

MagBLayer* bLayer::buildMagBLayer() const {
  if (mlayer == nullptr) {
    // If we have only one volume, do not build any MagBSector.
    if (sectors.empty()) {
      if (debug && size != 0) {
        std::cout << "ERROR: bLayer::buildMagBLayer, 0 sectors but " << size << " volumes" << std::endl;
      }
      // Technically we might have only one bSector built and we would
      // not need a separate MagBLayer constructor...
      mlayer = new MagBLayer(theVolumes.front()->magVolume, minR());
    }

    // If we have several sectors, create the MagBSector
    std::vector<MagBSector*> mSectors;
    for (unsigned int i = 0; i < sectors.size(); ++i) {
      mSectors.push_back(sectors[i].buildMagBSector());
    }
    mlayer = new MagBLayer(mSectors, minR());
  }
  return mlayer;
}
