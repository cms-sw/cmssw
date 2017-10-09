// #include "Utilities/Configuration/interface/Architecture.h"

/*
 *  See header file for a description of this class.
 *
 *  \author N. Amapane - INFN Torino
 */

#include "MagneticField/Layers/interface/MagBSlab.h"
#include "MagneticField/VolumeGeometry/interface/MagVolume.h"
#include "MagneticField/VolumeGeometry/interface/MagVolume6Faces.h"

#include "MagneticField/Layers/interface/MagVerbosity.h"
#include <iostream>

using namespace std;

MagBSlab::MagBSlab(const vector<MagVolume*>& volumes, double zMin) :
  theVolumes(volumes),
  theZMin(zMin)
{}

MagBSlab::~MagBSlab(){  
  for (vector<MagVolume *>::const_iterator ivol = theVolumes.begin();
       ivol != theVolumes.end(); ++ivol) {
    delete (*ivol);
  }
}


const MagVolume* MagBSlab::findVolume(const GlobalPoint & gp, double tolerance) const {
  for(vector<MagVolume*>::const_iterator ivol = theVolumes.begin();
	ivol != theVolumes.end(); ++ivol) {
    // FIXME : use a binfinder
    // TOFIX
    if (verbose::debugOut) cout << "        Trying volume "
			       << (static_cast<MagVolume6Faces*>(*ivol))->volumeNo << endl;
    if ( (*ivol)->inside(gp,tolerance) ) return (*ivol);
  }

  return 0;
}
