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
#include "RecoLocalTracker/SiStripRecHitConverter/interface/SiStripRecHitMatcher.h"

#include "DataFormats/TrackerRecHit2D/interface/SiStripRecHit2DLocalPosCollection.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripRecHit2DMatchedLocalPosCollection.h"
#include "DataFormats/SiStripCluster/interface/SiStripClusterCollection.h"
#include "Geometry/TrackerSimAlgo/interface/StripGeomDetUnit.h"
#include "Geometry/CommonDetUnit/interface/TrackingGeometry.h"
#include "MagneticField/Engine/interface/MagneticField.h"
class SiStripRecHitConverterAlgorithm 
{
 public:
  
  SiStripRecHitConverterAlgorithm(const edm::ParameterSet& conf);
  ~SiStripRecHitConverterAlgorithm();
  

  /// Runs the algorithm

    void run(const SiStripClusterCollection* input,SiStripRecHit2DMatchedLocalPosCollection & outmatched,SiStripRecHit2DLocalPosCollection & outrphi, SiStripRecHit2DLocalPosCollection & outstereo,const TrackingGeometry& tracker,const MagneticField &BField);
    void run(const SiStripClusterCollection* input, SiStripRecHit2DMatchedLocalPosCollection&  output, SiStripRecHit2DLocalPosCollection&  outrphi,SiStripRecHit2DLocalPosCollection&  outstereo,const TrackingGeometry & tracker,const MagneticField &BField,LocalVector trackdirection);
 private:

  edm::ParameterSet conf_;
  SiStripRecHitMatcher *clustermatch_;
};

#endif
