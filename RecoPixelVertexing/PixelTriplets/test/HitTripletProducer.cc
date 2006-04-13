#include <iostream>
#include <memory>
#include <string>

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

#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/TrackerGeometryBuilder/interface/PixelGeomDetUnit.h"
#include "Geometry/TrackerGeometryBuilder/interface/PixelGeomDetType.h"

#include "DataFormats/TrackerRecHit2D/interface/SiPixelRecHitCollection.h"

#include "RecoPixelVertexing/PixelTriplets/interface/OrderedHitTriplets.h"
#include "RecoTracker/TkTrackingRegions/interface/GlobalTrackingRegion.h"
#include "RecoPixelVertexing/PixelTriplets/interface/PixelHitTripletGenerator.h"


using namespace std;

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
  cout << " here HitTripletProducer CTOR" << endl;
  edm::LogInfo ("HitTripletProducer")<<"CTOR";
}

HitTripletProducer::~HitTripletProducer() 
{ 
  cout << " here HitTripletProducer DTOR" << endl;
  delete generator;
}

void HitTripletProducer::analyze(
    const edm::Event& ev, const edm::EventSetup& es)
{
  cout << " HERE begin of produce" << endl;


  cout << " ASKING FOR TRACKER!" << endl;
  edm::ESHandle<TrackerGeometry> tracker;
  es.get<TrackerDigiGeometryRecord>().get(tracker);
  cout << " AFTER ASKING FOR TRACKER! "  << endl;

  edm::Handle<SiPixelRecHitCollection> pixelHits;
  ev.getByType(pixelHits);

  cout << " before init" << endl;
  if (!generator) {
   generator = new PixelHitTripletGenerator;
  }
  generator->init(*pixelHits,es); 
  cout << " after init" << endl;

  GlobalTrackingRegion region;
  OrderedHitTriplets triplets;
  cout << " here 2, call for triplets" << endl;
  generator->hitTriplets(region,triplets,es);
  cout << " here 3, after call for triplets" << endl;
  cout << "size of triplets: " << triplets.size() << endl;
  cout << " HERE real end" << endl;

}
DEFINE_FWK_MODULE(HitTripletProducer)
