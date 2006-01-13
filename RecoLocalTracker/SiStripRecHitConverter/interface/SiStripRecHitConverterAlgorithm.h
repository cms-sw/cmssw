#ifndef SiStripRecHitConverterAlgorithm_h
#define SiStripRecHitConverterAlgorithm_h
# // -*-C++-*-
/** \class SiStripRecHitConverterAlgorithm
 *
 * SiStripRecHitConverterAlgorithm finds seeds
 *
 * \author Oliver Gutsche, Fermilab
 *
 * \version   1st Version Aug. 1, 2005
 *
 ************************************************************/

#include <string>

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "RecoLocalTracker/SiStripRecHitConverter/interface/SiStripClusterMatch.h"

#include "DataFormats/TrackerRecHit2D/interface/SiStripRecHit2DLocalPosCollection.h"
#include "DataFormats/SiStripCluster/interface/SiStripClusterCollection.h"

#include "Geometry/CommonDetUnit/interface/TrackingGeometry.h"

class SiStripRecHitConverterAlgorithm 
{
 public:
  
  SiStripRecHitConverterAlgorithm(const edm::ParameterSet& conf);
  ~SiStripRecHitConverterAlgorithm();
  

  /// Runs the algorithm
    void run(const SiStripClusterCollection* input, SiStripRecHit2DLocalPosCollection&  output, SiStripRecHit2DLocalPosCollection&  outrphi,SiStripRecHit2DLocalPosCollection&  outstereo,const TrackingGeometry & tracker);

 private:

  edm::ParameterSet conf_;
  SiStripClusterMatch *clustermatch_;
};

#endif
