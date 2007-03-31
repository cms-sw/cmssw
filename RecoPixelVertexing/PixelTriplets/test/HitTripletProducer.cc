#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "RecoTracker/TkTrackingRegions/interface/OrderedHitsGeneratorFactory.h"
#include "RecoTracker/TkTrackingRegions/interface/OrderedHitsGenerator.h"
#include "RecoTracker/TkTrackingRegions/interface/GlobalTrackingRegion.h"


class HitTripletProducer : public edm::EDAnalyzer {
public:
  explicit HitTripletProducer(const edm::ParameterSet& conf);
  ~HitTripletProducer();
  virtual void beginJob(const edm::EventSetup& iSetup);
  virtual void analyze(const edm::Event&, const edm::EventSetup&);
  virtual void endJob() { }
private:
  edm::ParameterSet theConfig;
  OrderedHitsGenerator * theGenerator;
};

HitTripletProducer::HitTripletProducer(const edm::ParameterSet& conf) 
  : theConfig(conf), theGenerator(0)
{
  edm::LogInfo("HitTripletProducer")<<" CTOR";
}

HitTripletProducer::~HitTripletProducer() 
{ 
  edm::LogInfo("HitTripletProducer")<<" DTOR";
  delete theGenerator;
}

void HitTripletProducer::beginJob(const edm::EventSetup& es)
{
  edm::ParameterSet orderedPSet =
      theConfig.getParameter<edm::ParameterSet>("OrderedHitsFactoryPSet");
  std::string orderedName = orderedPSet.getParameter<std::string>("ComponentName");
  theGenerator = OrderedHitsGeneratorFactory::get()->create( orderedName, orderedPSet);
}

void HitTripletProducer::analyze(
    const edm::Event& ev, const edm::EventSetup& es)
{

  GlobalTrackingRegion region;
  const OrderedSeedingHits & triplets = theGenerator->run(region,ev,es);
  edm::LogInfo("HitTripletProducer") << "size of triplets: "<<triplets.size();

}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(HitTripletProducer);
