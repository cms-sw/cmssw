// #include "Utilities/Configuration/interface/Architecture.h"

/*
 *  See header file for a description of this class.
 *
 *  \author N. Amapane - INFN Torino
 */

#include "MagneticField/Layers/interface/MagBRod.h"
#include "MagneticField/Layers/interface/MagBSlab.h"

#include "MagneticField/Layers/interface/MagVerbosity.h"

#include <iostream>

using namespace std;

MagBRod::MagBRod(vector<MagBSlab*>& slabs, Geom::Phi<float> phiMin) :
  theSlabs(slabs),
  thePhiMin(phiMin),
  theBinFinder(0)
{
  // TOFIX
//   if (verbose.debugOut) cout << "Building MagBRod with " << theSlabs.size()
// 		  << " slabs, minPhi " << thePhiMin << endl;
  
  if (theSlabs.size()>1) { // Set the binfinder
    vector<double> zBorders;
    for (vector<MagBSlab *>::const_iterator islab = theSlabs.begin();
	 islab != theSlabs.end(); ++islab) {
  // TOFIX
      if (verbose::debugOut) cout << (*islab)->minZ() <<endl;
      //FIXME assume layers are already sorted in Z
      zBorders.push_back((*islab)->minZ());
    }
    theBinFinder = new MagBinFinders::GeneralBinFinderInZ<double>(zBorders);
  }
}

MagBRod::~MagBRod() {
  delete theBinFinder;
  
  for (vector<MagBSlab *>::const_iterator islab = theSlabs.begin();
       islab != theSlabs.end(); ++islab) {
    delete (*islab);
  }
}

const MagVolume * MagBRod::findVolume(const GlobalPoint & gp, double tolerance) const {
  const MagVolume * result = 0;
  float Z = gp.z();

  int bin = 0;
  if (theBinFinder!=0) { // true if there is > 1 bin
    bin = theBinFinder->binIndex(Z);
  }
  
  // TOFIX
  if (verbose::debugOut) cout << "       Trying slab at Z " << theSlabs[bin]->minZ()
			      << " " << Z << endl ;
  result = theSlabs[bin]->findVolume(gp, tolerance);
  // TOFIX
  if (verbose::debugOut) cout << "***In guessed bslab"
			      << (result==0? " failed " : " OK ") <<endl;  

  return result;
}



