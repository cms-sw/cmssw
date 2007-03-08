#ifndef SiStripRecHitConverterAlgorithm_h
#define SiStripRecHitConverterAlgorithm_h
# // -*-C++-*-
/** \class SiStripRecHitConverterAlgorithm
 *
 * SiStripRecHitConverterAlgorithm convets clusters into rechits
 *
 * \author C. Genta
 *
 * \version   1st Version Aug. 1, 2005
 *
 ************************************************************/

#include <string>

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "RecoLocalTracker/SiStripRecHitConverter/interface/SiStripRecHitMatcher.h"
#include "RecoLocalTracker/ClusterParameterEstimator/interface/StripClusterParameterEstimator.h"

#include "DataFormats/TrackerRecHit2D/interface/SiStripRecHit2DCollection.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripMatchedRecHit2DCollection.h"
#include "DataFormats/SiStripCluster/interface/SiStripClusterCollection.h"
#include "DataFormats/Common/interface/DetSetVector.h"

#include "Geometry/TrackerGeometryBuilder/interface/StripGeomDetUnit.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"

#include "MagneticField/Engine/interface/MagneticField.h"

#include "DataFormats/Common/interface/Handle.h"

class SiStripRecHitConverterAlgorithm 
{
 public:
  
  SiStripRecHitConverterAlgorithm(const edm::ParameterSet& conf);
  ~SiStripRecHitConverterAlgorithm();
  

  /// Runs the algorithm

    void run(edm::Handle<edm::DetSetVector<SiStripCluster> >  input,SiStripMatchedRecHit2DCollection & outmatched,SiStripRecHit2DCollection & outrphi, SiStripRecHit2DCollection & outstereo,const TrackerGeometry& tracker,const StripClusterParameterEstimator &stripcpe, const SiStripRecHitMatcher &clustermatch_);
    void run(edm::Handle<edm::DetSetVector<SiStripCluster> >  input, SiStripMatchedRecHit2DCollection&  output, SiStripRecHit2DCollection&  outrphi,SiStripRecHit2DCollection&  outstereo,const TrackerGeometry & tracker,const StripClusterParameterEstimator &stripcpe ,const SiStripRecHitMatcher &clustermatch_, LocalVector trackdirection);
 private:

  edm::ParameterSet conf_;
};

#endif
