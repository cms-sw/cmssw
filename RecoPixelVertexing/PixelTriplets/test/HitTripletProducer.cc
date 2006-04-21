#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/TrackerRecHit2D/interface/SiPixelRecHitCollection.h"
#include "RecoPixelVertexing/PixelTriplets/interface/OrderedHitTriplets.h"
#include "RecoTracker/TkTrackingRegions/interface/GlobalTrackingRegion.h"
#include "RecoPixelVertexing/PixelTriplets/interface/PixelHitTripletGenerator.h"


class HitTripletProducer : public edm::EDAnalyzer {
public:
  explicit HitTripletProducer(const edm::ParameterSet& conf);
  ~HitTripletProducer();
  virtual void beginJob(const edm::EventSetup& iSetup) { }
  virtual void analyze(const edm::Event&, const edm::EventSetup&);
  virtual void endJob() { }
private:
  PixelHitTripletGenerator * generator;
};

HitTripletProducer::HitTripletProducer(const edm::ParameterSet& conf) 
  : generator(0)
{
  edm::LogInfo("HitTripletProducer")<<" CTOR";
}

HitTripletProducer::~HitTripletProducer() 
{ 
  edm::LogInfo("HitTripletProducer")<<" DTOR";
//  delete generator;
}

void HitTripletProducer::analyze(
    const edm::Event& ev, const edm::EventSetup& es)
{
  edm::Handle<SiPixelRecHitCollection> pixelHits;
  ev.getByType(pixelHits);

  generator = new PixelHitTripletGenerator;
  generator->init(*pixelHits,es); 

  GlobalTrackingRegion region;
  OrderedHitTriplets triplets;
  generator->hitTriplets(region,triplets,es);
  edm::LogInfo("HitTripletProducer") << "size of triplets: "<<triplets.size();
  delete generator;

}
DEFINE_FWK_MODULE(HitTripletProducer)
