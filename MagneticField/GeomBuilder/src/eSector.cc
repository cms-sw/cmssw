/*
 *  See header file for a description of this class.
 *
 *  \author N. Amapane - INFN Torino
 */

#include "eSector.h"
#include "printUniqueNames.h"
#include "Utilities/BinningTools/interface/ClusterizingHistogram.h"
#include "MagneticField/Layers/interface/MagESector.h"
#include "MagneticField/Layers/interface/MagVerbosity.h"
#include "Utilities/General/interface/precomputed_value_sort.h"

#include <algorithm>
#include <iostream>

using namespace SurfaceOrientation;
using namespace std;
using namespace magneticfield;

// The ctor is in charge of finding layers inside the sector.
eSector::eSector(handles::const_iterator begin, handles::const_iterator end, bool debugFlag)
    : theVolumes(begin, end), msector(nullptr), debug(debugFlag) {
  //FIXME!!!
  //precomputed_value_sort(theVolumes.begin(), theVolumes.end(), ExtractAbsZ());
  precomputed_value_sort(theVolumes.begin(), theVolumes.end(), ExtractZ());

  // Clusterize in Z
  const float resolution = 1.;  // cm //FIXME ??
  float zmin = theVolumes.front()->center().z() - resolution;
  float zmax = theVolumes.back()->center().z() + resolution;
  ClusterizingHistogram hisZ(int((zmax - zmin) / resolution) + 1, zmin, zmax);

  if (debug)
    cout << "     Z layers: " << zmin << " " << zmax << endl;

  handles::const_iterator first = theVolumes.begin();
  handles::const_iterator last = theVolumes.end();

  for (handles::const_iterator i = first; i != last; ++i) {
    hisZ.fill((*i)->center().z());
  }
  vector<float> zClust = hisZ.clusterize(resolution);

  if (debug)
    cout << "     Found " << zClust.size() << " clusters in Z, "
         << " layers: " << endl;

  handles::const_iterator layStart = first;
  handles::const_iterator separ = first;

  for (unsigned int i = 0; i < zClust.size() - 1; ++i) {
    float zSepar = (zClust[i] + zClust[i + 1]) / 2.f;
    while ((*separ)->center().z() < zSepar)
      ++separ;
    if (debug) {
      cout << "     Layer at: " << zClust[i] << " elements: " << separ - layStart << " unique volumes: ";
      printUniqueNames(layStart, separ);
    }

    layers.push_back(eLayer(layStart, separ));
    layStart = separ;
  }
  {
    if (debug) {
      cout << "     Layer at: " << zClust.back() << " elements: " << last - separ << " unique volumes: ";
      printUniqueNames(separ, last);
    }
    layers.push_back(eLayer(separ, last));
  }

  // FIXME: Check that all layers have the same dz?.
}

MagESector* eSector::buildMagESector() const {
  if (msector == nullptr) {
    vector<MagELayer*> mLayers;
    for (vector<eLayer>::const_iterator lay = layers.begin(); lay != layers.end(); ++lay) {
      mLayers.push_back((*lay).buildMagELayer());
    }
    msector = new MagESector(mLayers, theVolumes.front()->minPhi());  //FIXME
  }
  return msector;
}
