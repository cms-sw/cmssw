#ifndef PixelTracksMaker_H
#define PixelTracksMaker_H

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "RecoPixelVertexing/PixelTrackFitting/interface/TracksWithHits.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"

#include <string>

class PixelFitter;
class PixelTrackFilter;
class TrackingRegionProducer;
class MagneticField;
class TrackerGeometry;
class ParticlePropagator; 

namespace edm { 
  class ParameterSet;
  class Event;
  class EventSetup;
}


class PixelTracksMaker :  public edm::EDProducer {

public:
  explicit PixelTracksMaker(const edm::ParameterSet& conf);

  ~PixelTracksMaker();

  virtual void beginJob(const edm::EventSetup& es);

  virtual void produce(edm::Event& ev, const edm::EventSetup& es);

private:

  bool compatibleWithVertex(GlobalPoint& gpos1, GlobalPoint& gpos2); 

  edm::ParameterSet theConfig;
  const MagneticField*  theMagField;
  const TrackerGeometry*  theGeometry;

  double pTMin;
  double maxD0;
  double maxZ0;
  unsigned minRecHits;
  std::string hitProducer;
  double originRadius;
  double originHalfLength;
  double originpTMin;


  const PixelFitter       * theFitter;
  const PixelTrackFilter  * theFilter;
        TrackingRegionProducer* theRegionProducer;
};
#endif


