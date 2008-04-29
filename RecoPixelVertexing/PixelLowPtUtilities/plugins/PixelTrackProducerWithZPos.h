#ifndef PixelTrackProducerWithZPos_H
#define PixelTrackProducerWithZPos_H

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "RecoPixelVertexing/PixelTrackFitting/interface/TracksWithHits.h"

#include "TrackingTools/TransientTrack/interface/TransientTrackBuilder.h"
#include "TrackingTools/TransientTrack/interface/TransientTrack.h"
#include "TrackingTools/Records/interface/TransientTrackRecord.h"

#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"

class PixelFitter;
class PixelTrackCleaner;
class TrackHitsFilter;
class OrderedHitsGenerator;
class TrackingRegionProducer;
class Track;


namespace edm { class Event; class EventSetup; }

class PixelTrackProducerWithZPos :  public edm::EDProducer
{
  public:
    explicit PixelTrackProducerWithZPos(const edm::ParameterSet& conf);
    ~PixelTrackProducerWithZPos();
    virtual void produce(edm::Event& ev, const edm::EventSetup& es);
 
  private:
    void beginJob(const edm::EventSetup& es);
    std::pair<float,float> refitWithVertex (const reco::Track & recTrack,
                                       const reco::VertexCollection* vertices);

    void store(edm::Event& ev,
               const pixeltrackfitting::TracksWithRecHits & selectedTracks);

    edm::ParameterSet theConfig;

    const PixelFitter       * theFitter;
    const TrackHitsFilter   * theFilter;
          PixelTrackCleaner * theCleaner;  
          OrderedHitsGenerator * theGenerator;
          TrackingRegionProducer* theRegionProducer;

    const TransientTrackBuilder * theTTBuilder;
    bool theUseFoundVertices;
    bool theUseChi2Cut;

   double thePtMin, theOriginRadius;
};
#endif
