// #include "Utilities/Configuration/interface/Architecture.h"

/*
 *  See header file for a description of this class.
 *
 *  \author N. Amapane - INFN Torino
 */

#include "MagneticField/Layers/interface/MagESector.h"
#include "MagneticField/Layers/interface/MagELayer.h"

#include "MagneticField/Layers/interface/MagVerbosity.h"

#include <iostream>

using namespace std;

MagESector::MagESector(vector<MagELayer*>& layers, Geom::Phi<float> phiMin):
  theLayers(layers),
  thePhiMin(phiMin) 
{}


MagESector::~MagESector(){
  for (vector<MagELayer *>::const_iterator ilay = theLayers.begin();
       ilay != theLayers.end(); ++ilay) {
    delete (*ilay);
  }
}


const MagVolume * MagESector::findVolume(const GlobalPoint & gp, double tolerance) const {
  const MagVolume * result = 0;
  float Z = gp.z();

  // FIXME : use a binfinder
  for(vector<MagELayer*>::const_reverse_iterator ilay = theLayers.rbegin();
	ilay != theLayers.rend(); ++ilay) {

    if (Z+tolerance>(*ilay)->minZ()) {
      if (Z-tolerance<(*ilay)->maxZ()) {
#ifdef MF_DEBUG
	cout << "  Trying elayer at Z " << (*ilay)->minZ() << " " << Z << endl ;
#endif
      	result = (*ilay)->findVolume(gp, tolerance);
#ifdef MF_DEBUG
	cout << "***In elayer " << (result==0? " failed " : " OK ") << endl;
#endif
      } else {
	// break;  // FIXME: OK if sorted by maxZ
      }
    }
    if (result!=0) return result;
  }

  return 0;
}
