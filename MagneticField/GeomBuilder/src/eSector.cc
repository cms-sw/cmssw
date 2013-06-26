// #include "Utilities/Configuration/interface/Architecture.h"

/*
 *  See header file for a description of this class.
 *
 *  $Date: 2007/03/09 14:38:23 $
 *  $Revision: 1.5 $
 *  \author N. Amapane - INFN Torino
 */

#include "MagneticField/GeomBuilder/src/eSector.h"
#include "Utilities/BinningTools/interface/ClusterizingHistogram.h"
#include "MagneticField/Layers/interface/MagESector.h"
#include "MagneticField/Layers/interface/MagVerbosity.h"

#include <algorithm>
#include "Utilities/General/interface/precomputed_value_sort.h"

using namespace SurfaceOrientation;
using namespace std;

// The ctor is in charge of finding layers inside the sector.
MagGeoBuilderFromDDD::eSector::eSector(handles::const_iterator begin,
				       handles::const_iterator end) :
  theVolumes(begin,end),
  msector(0)
{
  //FIXME!!!
  //precomputed_value_sort(theVolumes.begin(), theVolumes.end(), ExtractAbsZ());
  precomputed_value_sort(theVolumes.begin(), theVolumes.end(), ExtractZ());
  

  // Clusterize in Z
  const float resolution = 1.; // cm //FIXME ??
  float zmin = theVolumes.front()->center().z()-resolution;
  float zmax = theVolumes.back()->center().z()+resolution;
  ClusterizingHistogram  hisZ( int((zmax-zmin)/resolution) + 1, zmin, zmax);

  if (MagGeoBuilderFromDDD::debug) cout << "     Z layers: " << zmin << " " << zmax << endl;

  handles::const_iterator first = theVolumes.begin();
  handles::const_iterator last = theVolumes.end();  

  for (handles::const_iterator i=first; i!=last; ++i){
    hisZ.fill((*i)->center().z());
  }
  vector<float> zClust = hisZ.clusterize(resolution);

  if (MagGeoBuilderFromDDD::debug) cout << "     Found " << zClust.size() << " clusters in Z, "
		  << " layers: " << endl;

  handles::const_iterator layStart = first;
  handles::const_iterator separ = first;

  for (unsigned int i=0; i<zClust.size() - 1; ++i) {
    float zSepar = (zClust[i] + zClust[i+1])/2.f;
    while ((*separ)->center().z() < zSepar) ++separ;
    if (MagGeoBuilderFromDDD::debug) {
      cout << "     Layer at: " << zClust[i]
	   << " elements: " << separ-layStart << " unique volumes: ";
      volumeHandle::printUniqueNames(layStart, separ);
    }
    
    layers.push_back(eLayer(layStart, separ));
    layStart = separ;
  }
  {
    if (MagGeoBuilderFromDDD::debug) {
      cout << "     Layer at: " << zClust.back() <<" elements: " << last-separ
	   << " unique volumes: ";
      volumeHandle::printUniqueNames(separ,last);
    }
    layers.push_back(eLayer(separ, last));
  }

  // FIXME: Check that all layers have the same dz?. 
  
}


MagGeoBuilderFromDDD::eSector::~eSector(){}


MagESector* MagGeoBuilderFromDDD::eSector::buildMagESector() const{
  if (msector==0) {
    vector<MagELayer*> mLayers;
    for (vector<eLayer>::const_iterator lay = layers.begin();
	 lay!=layers.end(); ++lay) {
      mLayers.push_back((*lay).buildMagELayer());
    }
    msector = new MagESector(mLayers, theVolumes.front()->minPhi()); //FIXME
  }
  return msector;
}
