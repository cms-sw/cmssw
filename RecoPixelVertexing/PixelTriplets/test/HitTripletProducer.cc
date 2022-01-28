#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "RecoTracker/TkTrackingRegions/interface/OrderedHitsGeneratorFactory.h"
#include "RecoTracker/TkTrackingRegions/interface/OrderedHitsGenerator.h"
// #include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"

#include "RecoTracker/TkTrackingRegions/interface/TrackingRegionProducerFactory.h"
#include "RecoTracker/TkTrackingRegions/interface/TrackingRegionProducer.h"
#include "RecoTracker/TkTrackingRegions/interface/TrackingRegion.h"
#include "RecoTracker/TkTrackingRegions/interface/GlobalTrackingRegion.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"

//#include "FWCore/Framework/interface/ESWatcher.h"
//#include "UserCode/konec/test/R2DTimerObserver.h"
#include "TH1D.h"
#include "TFile.h"

class HitTripletProducer : public edm::one::EDAnalyzer<> {
public:
  explicit HitTripletProducer(const edm::ParameterSet& conf);
  ~HitTripletProducer();
  virtual void analyze(const edm::Event&, const edm::EventSetup&);

private:
  std::unique_ptr<OrderedHitsGenerator> theGenerator;
  std::unique_ptr<TrackingRegionProducer> theRegionProducer;
  TH1D *hCPU, *hNum;
};

HitTripletProducer::HitTripletProducer(const edm::ParameterSet& conf) {
  edm::LogInfo("HitTripletProducer") << " CTOR";
  hCPU = new TH1D("hCPU", "hCPU", 140, 0., 0.070);
  hNum = new TH1D("hNum", "hNum", 250, 0., 500.);

  edm::ParameterSet orderedPSet = conf.getParameter<edm::ParameterSet>("OrderedHitsFactoryPSet");
  std::string orderedName = orderedPSet.getParameter<std::string>("ComponentName");
  edm::ConsumesCollector iC = consumesCollector();
  theGenerator = OrderedHitsGeneratorFactory::get()->create(orderedName, orderedPSet, iC);

  edm::ParameterSet regfactoryPSet = conf.getParameter<edm::ParameterSet>("RegionFactoryPSet");
  std::string regfactoryName = regfactoryPSet.getParameter<std::string>("ComponentName");
  theRegionProducer = TrackingRegionProducerFactory::get()->create(regfactoryName, regfactoryPSet, consumesCollector());
}

HitTripletProducer::~HitTripletProducer() {
  edm::LogInfo("HitTripletProducer") << " DTOR";

  TFile rootFile("analysis.root", "RECREATE", "my histograms");
  hCPU->Write();
  hNum->Write();
  rootFile.Close();
}

void HitTripletProducer::analyze(const edm::Event& ev, const edm::EventSetup& es) {
  typedef std::vector<std::unique_ptr<TrackingRegion> > Regions;
  Regions regions = theRegionProducer->regions(ev, es);
  const TrackingRegion& region = *regions[0];

  //  static R2DTimerObserver timer("**** MY TIMING REPORT ***");
  //  timer.start();
  edm::LogInfo("HitTripletProducer") << "call triplets! ";
  const OrderedSeedingHits& triplets = theGenerator->run(region, ev, es);
  //  timer.stop();
  //  hCPU->Fill( timer.lastMeasurement().real() );
  hNum->Fill(triplets.size());
  edm::LogInfo("HitTripletProducer") << "size of triplets: " << triplets.size();

  theGenerator->clear();
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(HitTripletProducer);
