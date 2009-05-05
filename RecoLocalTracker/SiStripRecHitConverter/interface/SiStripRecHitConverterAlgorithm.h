#ifndef SiStripRecHitConverterAlgorithm_h
#define SiStripRecHitConverterAlgorithm_h

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/DetSetVectorNew.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripRecHit2DCollection.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripMatchedRecHit2DCollection.h"
#include "DataFormats/GeometryVector/interface/LocalVector.h"

class SiStripRecHitMatcher;
class StripClusterParameterEstimator;
class RefGetter;
class StripGeomDetUnit;
class TrackerGeometry;
class SiStripQuality;


class SiStripRecHitConverterAlgorithm 
{
 public:
  
  SiStripRecHitConverterAlgorithm(const edm::ParameterSet& conf) : conf_(conf) {}
  
  void run(edm::Handle<edmNew::DetSetVector<SiStripCluster> > input,
	   SiStripMatchedRecHit2DCollection & outmatched,
	   SiStripRecHit2DCollection & outrphi, 
	   SiStripRecHit2DCollection & outstereo, 
	   SiStripRecHit2DCollection & outrphiUnmatched, 
	   SiStripRecHit2DCollection & outstereoUnmatched,
	   const TrackerGeometry& tracker,const StripClusterParameterEstimator &stripcpe, 
	   const SiStripRecHitMatcher &clustermatch_, 
	   const SiStripQuality *quality = 0);

  void run(edm::Handle<edmNew::DetSetVector<SiStripCluster> >  input, 
	   SiStripMatchedRecHit2DCollection&  output, 
	   SiStripRecHit2DCollection&  outrphi,
	   SiStripRecHit2DCollection & outstereo, 
	   SiStripRecHit2DCollection & outrphiUnmatched, 
	   SiStripRecHit2DCollection & outstereoUnmatched,
	   const TrackerGeometry & tracker,
	   const StripClusterParameterEstimator &stripcpe ,
	   const SiStripRecHitMatcher &clustermatch_, 
	   LocalVector trackdirection, 
	   const SiStripQuality *quality = 0);

  void run(edm::Handle<edm::RefGetter<SiStripCluster> >  input1, 
	   edm::Handle<edm::LazyGetter<SiStripCluster> > input2, 
	   SiStripMatchedRecHit2DCollection & outmatched,
	   SiStripRecHit2DCollection & outrphi, 
	   SiStripRecHit2DCollection & outstereo, 
	   SiStripRecHit2DCollection & outrphiUnmatched, 
	   SiStripRecHit2DCollection & outstereoUnmatched,
	   const TrackerGeometry& tracker,
	   const StripClusterParameterEstimator &parameterestimator, 
	   const SiStripRecHitMatcher & matcher, 
	   const SiStripQuality *quality = 0);
  
 private:

  edm::ParameterSet conf_;
  
  void match(SiStripMatchedRecHit2DCollection & outmatched,
	     SiStripRecHit2DCollection & outrphi, 
	     SiStripRecHit2DCollection & outstereo, 
	     SiStripRecHit2DCollection & outrphiUnmatched, 
	     SiStripRecHit2DCollection & outstereoUnmatched,
	     const TrackerGeometry& tracker, 
	     const SiStripRecHitMatcher & matcher,
	     LocalVector trackdirection) const;
  
  void fillBad128StripBlocks(const SiStripQuality &quality, 
			     const uint32_t &detid, 
			     bool bad128StripBlocks[6]) const;
  
  bool isMasked(const SiStripCluster &cluster, 
		bool bad128StripBlocks[6]) const;
    
};

#endif
