/* \file
 *  See header file for a description of this class.
 *
 *  \author N. Amapane - INFN Torino
 */

#include "bSlab.h"
#include "MagneticField/VolumeGeometry/interface/MagVolume6Faces.h"
#include "MagneticField/Layers/interface/MagBSlab.h"
#include "MagneticField/Layers/interface/MagVerbosity.h"

#include "Utilities/General/interface/precomputed_value_sort.h"

#include <iostream>

using namespace SurfaceOrientation;
using namespace std;
using namespace magneticfield;

bSlab::bSlab(handles::const_iterator begin, handles::const_iterator end, bool debugVal)
    : volumes(begin, end), mslab(nullptr), debug(debugVal) {
  if (volumes.size() > 1) {
    // Sort volumes by dphi i.e. phi(j)-phi(i) > 0 if j>1.
    precomputed_value_sort(volumes.begin(), volumes.end(), ExtractPhiMax(), LessDPhi());

    if (debug)
      cout << "        Slab has " << volumes.size() << " volumes" << endl;

    // Check that all volumes have the same dZ
    handles::const_iterator i = volumes.begin();
    float Zmax = (*i)->surface(zplus).position().z();
    float Zmin = (*i)->surface(zminus).position().z();
    for (++i; i != volumes.end(); ++i) {
      const float epsilon = 0.001;
      if (fabs(Zmax - (*i)->surface(zplus).position().z()) > epsilon ||
          fabs(Zmin - (*i)->surface(zminus).position().z()) > epsilon) {
        if (debug)
          cout << "*** WARNING: slabs Z coords not matching: D_Zmax = "
               << fabs(Zmax - (*i)->surface(zplus).position().z())
               << " D_Zmin = " << fabs(Zmin - (*i)->surface(zminus).position().z()) << endl;
      }
    }
  }
}

Geom::Phi<float> bSlab::minPhi() const { return volumes.front()->minPhi(); }

Geom::Phi<float> bSlab::maxPhi() const { return volumes.back()->maxPhi(); }

MagBSlab* bSlab::buildMagBSlab() const {
  if (mslab == nullptr) {
    vector<MagVolume*> mVols;
    for (handles::const_iterator vol = volumes.begin(); vol != volumes.end(); ++vol) {
      mVols.push_back((*vol)->magVolume);
    }
    mslab = new MagBSlab(mVols, volumes.front()->surface(zminus).position().z());  //FIXME
  }
  return mslab;
}
