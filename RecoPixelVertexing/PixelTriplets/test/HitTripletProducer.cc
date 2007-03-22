#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/TrackerRecHit2D/interface/SiPixelRecHitCollection.h"
#include "RecoPixelVertexing/PixelTriplets/interface/OrderedHitTriplets.h"
#include "RecoTracker/TkTrackingRegions/interface/GlobalTrackingRegion.h"
#include "RecoPixelVertexing/PixelTriplets/interface/CombinedHitTripletGenerator.h"


class HitTripletProducer : public edm::EDAnalyzer {
public:
  explicit HitTripletProducer(const edm::ParameterSet& conf);
  ~HitTripletProducer();
  virtual void beginJob(const edm::EventSetup& iSetup) { }
  virtual void analyze(const edm::Event&, const edm::EventSetup&);
  virtual void endJob() { }
private:
  edm::ParameterSet theConfig;
  CombinedHitTripletGenerator * generator;
};

HitTripletProducer::HitTripletProducer(const edm::ParameterSet& conf) 
  : theConfig(conf), generator(0)
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

//    edm::ParameterSet pset = theConfig.getParameter<edm::ParameterSet>("TripletsPSet");
//    XXXPixelTrackFilter * aGen =
//    XXXPixelTrackFilterFactory::get()->create("XXXPixelTrackFilterByKinematics",pset);
//    aGen->koko();


  if (!generator) {
    edm::ParameterSet pset = theConfig.getParameter<edm::ParameterSet>("TripletsPSet");
    generator = new CombinedHitTripletGenerator(pset);
  }

  GlobalTrackingRegion region;
  OrderedHitTriplets triplets;
  generator->hitTriplets(region,triplets,ev,es);
  edm::LogInfo("HitTripletProducer") << "size of triplets: "<<triplets.size();
  delete generator; generator=0;

}
DEFINE_FWK_MODULE(HitTripletProducer);
