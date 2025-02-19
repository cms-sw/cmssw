// #include "Utilities/Configuration/interface/Architecture.h"

/*
 *  See header file for a description of this class.
 *
 *  $Date: 2007/02/03 16:19:08 $
 *  $Revision: 1.3 $
 *  \author N. Amapane - INFN Torino
 */

#include "MagneticField/GeomBuilder/src/eLayer.h"
#include "MagneticField/VolumeGeometry/interface/MagVolume6Faces.h"
#include "MagneticField/Layers/interface/MagELayer.h"

#include "Utilities/General/interface/precomputed_value_sort.h"

using namespace SurfaceOrientation;
using namespace std;

//The ctor is in charge of finding sectors inside the layer.
MagGeoBuilderFromDDD::eLayer::eLayer(handles::const_iterator begin,
					handles::const_iterator end) :
  theVolumes(begin,end),
  mlayer(0) 
{
  //  bool debug=MagGeoBuilderFromDDD::debug;

  // Sort in R  
  precomputed_value_sort(theVolumes.begin(), theVolumes.end(), ExtractR());

//   if (debug) {
//     cout << " elements: " << theVolumes.size() << " unique volumes: ";
//     volumeHandle::printUniqueNames(theVolumes.begin(), theVolumes.end());
//   }
}

MagGeoBuilderFromDDD::eLayer::~eLayer(){}

// double MagGeoBuilderFromDDD::eLayer::minR() const {
//   // ASSUMPTION: a layer is only 1 volume thick (by construction). 
//   return theVolumes.front()->minR();
// }

// double MagGeoBuilderFromDDD::eLayer::maxR() const {
//   // ASSUMPTION: a layer is only 1 volume thick (by construction). 
//   return theVolumes.front()->maxR();
// }

MagELayer * MagGeoBuilderFromDDD::eLayer::buildMagELayer() const {
  if (mlayer==0) {
    //FIXME not guaranteed that all volumes in layer have the same zmin
    // and zmax!
    double zmin = 1e19;
    double zmax = -1e19;
    vector<MagVolume*> mVols;
    for (handles::const_iterator vol = theVolumes.begin();
	 vol!=theVolumes.end(); ++vol) {
      mVols.push_back((*vol)->magVolume);
      zmin = min(zmin, (*vol)->minZ());
      zmax = max(zmax, (*vol)->maxZ());
    }
    mlayer = new MagELayer(mVols, zmin, zmax);
  }
  return mlayer;
}

