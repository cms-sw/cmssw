// #include "Utilities/Configuration/interface/Architecture.h"

/*
 *  See header file for a description of this class.
 *
 *  \author N. Amapane - INFN Torino
 */

#include "MagneticField/Layers/interface/MagELayer.h"
#include "MagneticField/VolumeGeometry/interface/MagVolume.h"
#include "MagneticField/VolumeGeometry/interface/MagVolume6Faces.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <iostream>

using namespace std;

MagELayer::MagELayer(const vector<MagVolume*>& volumes, double zMin, double zMax)
    : theVolumes(volumes), theZMin(zMin), theZMax(zMax) {}

MagELayer::~MagELayer() {
  for (auto theVolume : theVolumes) {
    delete theVolume;
  }
}

const MagVolume* MagELayer::findVolume(const GlobalPoint& gp, double tolerance) const {
  for (auto theVolume : theVolumes) {
    // FIXME : use a binfinder
#ifdef EDM_ML_DEBUG
    {
      MagVolume6Faces* mv = static_cast<MagVolume6Faces*>(*ivol);
      LogTrace("MagGeometry") << "        Trying volume " << mv->volumeNo << " " << int(mv->copyno) << endl;
    }
#endif
    if (theVolume->inside(gp, tolerance))
      return theVolume;
  }

  return nullptr;
}
