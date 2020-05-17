// #include "Utilities/Configuration/interface/Architecture.h"

/*
 *  See header file for a description of this class.
 *
 *  \author N. Amapane - INFN Torino
 */

#include "MagneticField/Layers/interface/MagBSector.h"
#include "MagneticField/Layers/interface/MagBRod.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <iostream>

using namespace std;

MagBSector::MagBSector(vector<MagBRod*>& rods, Geom::Phi<float> phiMin) : theRods(rods), thePhiMin(phiMin) {}

MagBSector::~MagBSector() {
  for (auto theRod : theRods) {
    delete theRod;
  }
}

const MagVolume* MagBSector::findVolume(const GlobalPoint& gp, double tolerance) const {
  const MagVolume* result = nullptr;
  Geom::Phi<float> phi = gp.phi();

  // FIXME : use a binfinder
  for (auto theRod : theRods) {
    LogTrace("MagGeometry") << "     Trying rod at phi " << theRod->minPhi() << " " << phi << endl;
    result = theRod->findVolume(gp, tolerance);
    if (result != nullptr)
      return result;
  }

  return nullptr;
}
