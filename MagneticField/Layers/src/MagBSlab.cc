// #include "Utilities/Configuration/interface/Architecture.h"

/*
 *  See header file for a description of this class.
 *
 *  \author N. Amapane - INFN Torino
 */

#include "MagneticField/Layers/interface/MagBSlab.h"
#include "MagneticField/VolumeGeometry/interface/MagVolume.h"
#include "MagneticField/VolumeGeometry/interface/MagVolume6Faces.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <iostream>

using namespace std;

MagBSlab::MagBSlab(const vector<MagVolume*>& volumes, double zMin) : theVolumes(volumes), theZMin(zMin) {}

MagBSlab::~MagBSlab() {
  for (auto theVolume : theVolumes) {
    delete theVolume;
  }
}

const MagVolume* MagBSlab::findVolume(const GlobalPoint& gp, double tolerance) const {
  for (auto theVolume : theVolumes) {
    // FIXME : use a binfinder
    LogTrace("MagGeometry") << "        Trying volume " << (static_cast<MagVolume6Faces*>(theVolume))->volumeNo << endl;
    if (theVolume->inside(gp, tolerance))
      return theVolume;
  }

  return nullptr;
}
