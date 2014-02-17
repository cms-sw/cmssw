// #include "Utilities/Configuration/interface/Architecture.h"

/*
 *  See header file for a description of this class.
 *
 *  $Date: 2007/02/03 16:15:59 $
 *  $Revision: 1.2 $
 *  \author N. Amapane - INFN Torino
 */

#include "MagneticField/Layers/interface/MagESector.h"
#include "MagneticField/Layers/interface/MagELayer.h"

// #include "MagneticField/MagLayers/interface/MagVerbosity.h"

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


MagVolume * MagESector::findVolume(const GlobalPoint & gp, double tolerance) const {
  MagVolume * result = 0;
  float Z = gp.z();

  //  int count = 0;

  // FIXME : use a binfinder
  for(vector<MagELayer*>::const_reverse_iterator ilay = theLayers.rbegin();
	ilay != theLayers.rend(); ++ilay) {

    if (Z+tolerance>(*ilay)->minZ()) {
      if (Z-tolerance<(*ilay)->maxZ()) {
// 	if (verbose.debugOut) cout << "  Trying layer at Z " << (*ilay)->minZ()
// 			<< " " << Z << endl ;
	result = (*ilay)->findVolume(gp, tolerance);
// 	if (verbose.debugOut) {
// 	  cout << "***In elayer " << count << " " 
// 	       << (result==0? " failed " : " OK ") <<endl;
// 	  ++count;
// 	}
      } else {
	// break;  // FIXME: OK if sorted by maxZ
      }
    }
    if (result!=0) return result;
  }

  return 0;
}
