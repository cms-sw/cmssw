/*
 *  See header file for a description of this class.
 *
 *  \author N. Amapane - INFN Torino
 */

#include "eLayer.h"
#include "MagneticField/VolumeGeometry/interface/MagVolume6Faces.h"
#include "MagneticField/Layers/interface/MagELayer.h"

#include "Utilities/General/interface/precomputed_value_sort.h"

using namespace SurfaceOrientation;
using namespace std;
using namespace magneticfield;

//The ctor is in charge of finding sectors inside the layer.
eLayer::eLayer(handles::const_iterator begin, handles::const_iterator end) : theVolumes(begin, end), mlayer(nullptr) {
  //  bool debug=MagGeoBuilderFromDDD::debug;

  // Sort in R
  precomputed_value_sort(theVolumes.begin(), theVolumes.end(), ExtractR());

  //   if (debug) {
  //     cout << " elements: " << theVolumes.size() << " unique volumes: ";
  //     volumeHandle::printUniqueNames(theVolumes.begin(), theVolumes.end());
  //   }
}

// double MagGeoBuilderFromDDD::eLayer::minR() const {
//   // ASSUMPTION: a layer is only 1 volume thick (by construction).
//   return theVolumes.front()->minR();
// }

// double MagGeoBuilderFromDDD::eLayer::maxR() const {
//   // ASSUMPTION: a layer is only 1 volume thick (by construction).
//   return theVolumes.front()->maxR();
// }

MagELayer* eLayer::buildMagELayer() const {
  if (mlayer == nullptr) {
    //FIXME not guaranteed that all volumes in layer have the same zmin
    // and zmax!
    double zmin = 1e19;
    double zmax = -1e19;
    vector<MagVolume*> mVols;
    for (handles::const_iterator vol = theVolumes.begin(); vol != theVolumes.end(); ++vol) {
      mVols.push_back((*vol)->magVolume);
      zmin = min(zmin, (*vol)->minZ());
      zmax = max(zmax, (*vol)->maxZ());
    }
    mlayer = new MagELayer(mVols, zmin, zmax);
  }
  return mlayer;
}
