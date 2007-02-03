// #include "Utilities/Configuration/interface/Architecture.h"

/*
 *  See header file for a description of this class.
 *
 *  $Date: 2005/09/06 15:49:19 $
 *  $Revision: 1.1 $
 *  \author N. Amapane - INFN Torino
 */

#include "MagneticField/Layers/interface/MagELayer.h"
#include "MagneticField/VolumeGeometry/interface/MagVolume.h"
#include "MagneticField/VolumeGeometry/interface/MagVolume6Faces.h"

// #include "MagneticField/MagLayers/interface/MagVerbosity.h"
#include <iostream>

using namespace std;


MagELayer::MagELayer(vector<MagVolume*> volumes, double zMin, double zMax) :
  theVolumes(volumes),
  theZMin(zMin),
  theZMax(zMax)
{}

MagELayer::~MagELayer(){
  for (vector<MagVolume *>::const_iterator ivol = theVolumes.begin();
       ivol != theVolumes.end(); ++ivol) {
    delete (*ivol);
  }
}


MagVolume * 
MagELayer::findVolume(const GlobalPoint & gp, double tolerance) const {
  for(vector<MagVolume*>::const_iterator ivol = theVolumes.begin();
	ivol != theVolumes.end(); ++ivol) {
    // FIXME : use a binfinder
    // TOFIX
//     if (verbose.debugOut) cout << "        Trying volume "
// 		    << (static_cast<MagVolume6Faces*> (*ivol))->name << endl;
    if ( (*ivol)->inside(gp,tolerance) ) return (*ivol);
  }

  return 0;
}

