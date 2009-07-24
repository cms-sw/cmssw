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
#include "DataFormats/Common/interface/DetSetVectorNew.h"
#include "DataFormats/Common/interface/RefGetter.h"

#include "Geometry/TrackerGeometryBuilder/interface/StripGeomDetUnit.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"

#include "MagneticField/Engine/interface/MagneticField.h"

#include "DataFormats/Common/interface/Handle.h"
#include "CalibFormats/SiStripObjects/interface/SiStripQuality.h"

class SiStripRecHitConverterAlgorithm 
{
 public:
  
  SiStripRecHitConverterAlgorithm(const edm::ParameterSet& conf);
  ~SiStripRecHitConverterAlgorithm();
  

  /// Runs the algorithm

    void run(edm::Handle<edmNew::DetSetVector<SiStripCluster> >  input,SiStripMatchedRecHit2DCollection & outmatched,SiStripRecHit2DCollection & outrphi, SiStripRecHit2DCollection & outstereo, SiStripRecHit2DCollection & outrphiUnmatched, SiStripRecHit2DCollection & outstereoUnmatched,const TrackerGeometry& tracker,const StripClusterParameterEstimator &stripcpe, const SiStripRecHitMatcher &clustermatch_, const SiStripQuality *quality = 0);
    void run(edm::Handle<edmNew::DetSetVector<SiStripCluster> >  input, SiStripMatchedRecHit2DCollection&  output, SiStripRecHit2DCollection&  outrphi,SiStripRecHit2DCollection & outstereo, SiStripRecHit2DCollection & outrphiUnmatched, SiStripRecHit2DCollection & outstereoUnmatched,const TrackerGeometry & tracker,const StripClusterParameterEstimator &stripcpe ,const SiStripRecHitMatcher &clustermatch_, LocalVector trackdirection, const SiStripQuality *quality = 0);
    void run(edm::Handle<edm::RefGetter<SiStripCluster> >  input1, edm::Handle<edm::LazyGetter<SiStripCluster> > input2, SiStripMatchedRecHit2DCollection & outmatched,SiStripRecHit2DCollection & outrphi, SiStripRecHit2DCollection & outstereo, SiStripRecHit2DCollection & outrphiUnmatched, SiStripRecHit2DCollection & outstereoUnmatched,const TrackerGeometry& tracker,const StripClusterParameterEstimator &parameterestimator, const SiStripRecHitMatcher & matcher, const SiStripQuality *quality = 0);

 private:

    //void convert(const SiStripCluster& cluster,SiStripRecHit2DCollection & outrphi, SiStripRecHit2DCollection & outstereo,const TrackerGeometry& tracker,const StripClusterParameterEstimator &parameterestimator);
    
    void match(SiStripMatchedRecHit2DCollection & outmatched,SiStripRecHit2DCollection & outrphi, SiStripRecHit2DCollection & outstereo, SiStripRecHit2DCollection & outrphiUnmatched, SiStripRecHit2DCollection & outstereoUnmatched,const TrackerGeometry& tracker, const SiStripRecHitMatcher & matcher,LocalVector trackdirection) const;
      
  edm::ParameterSet conf_;

    inline bool isMasked(const SiStripCluster &cluster, bool bad128StripBlocks[6]) const {
        if ( bad128StripBlocks[cluster.firstStrip() >> 7] ) {
            if ( bad128StripBlocks[(cluster.firstStrip()+cluster.amplitudes().size())  >> 7] ||
                 bad128StripBlocks[static_cast<int32_t>(cluster.barycenter()-0.499999) >> 7] ) {
                return true;
            }
        } else {
            if ( bad128StripBlocks[(cluster.firstStrip()+cluster.amplitudes().size())  >> 7] &&
                 bad128StripBlocks[static_cast<int32_t>(cluster.barycenter()-0.499999) >> 7] ) {
                return true;
            }
        }
        return false;
    }
    void   fillBad128StripBlocks(const SiStripQuality &quality, const uint32_t &detid, bool bad128StripBlocks[6]) const ;
};

#endif
