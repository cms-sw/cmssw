// #include "Utilities/Configuration/interface/Architecture.h"

/*
 *  See header file for a description of this class.
 *
 *  \author N. Amapane - INFN Torino
 */

#include "MagneticField/Layers/interface/MagBRod.h"
#include "MagneticField/Layers/interface/MagBSlab.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <iostream>

using namespace std;

MagBRod::MagBRod(vector<MagBSlab *> &slabs, Geom::Phi<float> phiMin)
    : theSlabs(slabs), thePhiMin(phiMin), theBinFinder(nullptr) {
  //   LogTrace("MagGeometry") << "Building MagBRod with " << theSlabs.size()
  // 		  << " slabs, minPhi " << thePhiMin << endl;

  if (theSlabs.size() > 1) {  // Set the binfinder
    vector<double> zBorders;
    for (vector<MagBSlab *>::const_iterator islab = theSlabs.begin(); islab != theSlabs.end(); ++islab) {
      LogTrace("MagGeoBuilder") << " MagBSlab minZ=" << (*islab)->minZ() << endl;
      //FIXME assume layers are already sorted in Z
      zBorders.push_back((*islab)->minZ());
    }
    theBinFinder = new MagBinFinders::GeneralBinFinderInZ<double>(zBorders);
  }
}

MagBRod::~MagBRod() {
  delete theBinFinder;

  for (vector<MagBSlab *>::const_iterator islab = theSlabs.begin(); islab != theSlabs.end(); ++islab) {
    delete (*islab);
  }
}

const MagVolume *MagBRod::findVolume(const GlobalPoint &gp, double tolerance) const {
  const MagVolume *result = nullptr;
  float Z = gp.z();

  int bin = 0;
  if (theBinFinder != nullptr) {  // true if there is > 1 bin
    bin = theBinFinder->binIndex(Z);
  }

  LogTrace("MagGeometry") << "       Trying slab at Z " << theSlabs[bin]->minZ() << " " << Z << endl;
  result = theSlabs[bin]->findVolume(gp, tolerance);
  LogTrace("MagGeometry") << "***In guessed bslab" << (result == nullptr ? " failed " : " OK ") << endl;

  return result;
}
