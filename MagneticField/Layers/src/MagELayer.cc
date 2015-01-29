// #include "Utilities/Configuration/interface/Architecture.h"

/*
 *  See header file for a description of this class.
 *
 *  \author N. Amapane - INFN Torino
 */

#include "MagneticField/Layers/interface/MagELayer.h"
#include "MagneticField/VolumeGeometry/interface/MagVolume.h"
#include "MagneticField/VolumeGeometry/interface/MagVolume6Faces.h"

#include "MagneticField/Layers/interface/MagVerbosity.h"
#include <iostream>

using namespace std;


MagELayer::MagELayer(const vector<MagVolume*>& volumes, double zMin, double zMax) :
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


const MagVolume * 
MagELayer::findVolume(const GlobalPoint & gp, double tolerance) const {
  for(vector<MagVolume*>::const_iterator ivol = theVolumes.begin();
	ivol != theVolumes.end(); ++ivol) {
    // FIXME : use a binfinder
#ifdef MF_DEBUG
    {
      MagVolume6Faces* mv = static_cast<MagVolume6Faces*> (*ivol);
      cout << "        Trying volume " << mv->volumeNo << " " << int(mv->copyno) << endl;
    }
#endif
    if ( (*ivol)->inside(gp,tolerance) ) return (*ivol);
  }

  return 0;
}

