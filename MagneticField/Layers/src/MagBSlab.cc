// #include "Utilities/Configuration/interface/Architecture.h"

/*
 *  See header file for a description of this class.
 *
 *  $Date: 2005/09/06 15:49:19 $
 *  $Revision: 1.1 $
 *  \author N. Amapane - INFN Torino
 */

#include "MagneticField/Layers/interface/MagBSlab.h"
#include "MagneticField/VolumeGeometry/interface/MagVolume.h"
#include "MagneticField/VolumeGeometry/interface/MagVolume6Faces.h"

#include "MagneticField/Layers/interface/MagVerbosity.h"
#include <iostream>

using namespace std;

MagBSlab::MagBSlab(vector<MagVolume*> volumes, double zMin) :
  theVolumes(volumes),
  theZMin(zMin)
{}

MagBSlab::~MagBSlab(){  
  for (vector<MagVolume *>::const_iterator ivol = theVolumes.begin();
       ivol != theVolumes.end(); ivol++) {
    delete (*ivol);
  }
}


MagVolume* MagBSlab::findVolume(const GlobalPoint & gp, double tolerance) const {
  for ( vector<MagVolume*>::const_iterator ivol = theVolumes.begin();
	ivol != theVolumes.end(); ivol++) {
    // FIXME : use a binfinder
    // TOFIX
    if (verbose::debugOut) cout << "        Trying volume "
			       << (static_cast<MagVolume6Faces*>(*ivol))->name << endl;
    if ( (*ivol)->inside(gp,tolerance) ) return (*ivol);
  }

  return 0;
}
