#ifndef TRACKERMUFILTER_H
#define TRACKERMUFILTER_H


#include "FWCore/Framework/interface/EDFilter.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "DataFormats/SiStripCluster/interface/SiStripCluster.h"
#include "DataFormats/DetId/interface/DetId.h"



#include "RecoLocalTracker/SiStripRecHitConverter/test/ValHit.h"

#include "Geometry/Vector/interface/LocalPoint.h"
#include "Geometry/Vector/interface/GlobalPoint.h"

#include "Geometry/CommonDetUnit/interface/GeomDetType.h"
#include "Geometry/CommonDetUnit/interface/GeomDetUnit.h"
#include "Geometry/TrackerGeometryBuilder/interface/GluedGeomDet.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/TrackerNumberingBuilder/interface/GeometricDet.h"
#include "Geometry/CommonTopologies/interface/PixelTopology.h"
#include "Geometry/CommonTopologies/interface/StripTopology.h"
#include "Geometry/TrackerGeometryBuilder/interface/PixelGeomDetType.h"
#include "Geometry/TrackerGeometryBuilder/interface/StripGeomDetType.h"
#include "Geometry/TrackerGeometryBuilder/interface/StripGeomDetUnit.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "Geometry/DTGeometry/interface/DTLayer.h"
#include "Geometry/DTGeometry/interface/DTGeometry.h"
#include "Geometry/Records/interface/MuonGeometryRecord.h"
#include "DataFormats/DTDigi/interface/DTDigiCollection.h"

//Added by Max
#include "Geometry/DTGeometry/interface/DTChamber.h"
#include "Geometry/DTGeometry/interface/DTLayer.h"
#include "Geometry/DTGeometry/interface/DTLayerType.h"
#include "Geometry/DTGeometry/interface/DTSuperLayer.h"
#include "Geometry/DTGeometry/interface/DTTopology.h"

#include "Geometry/Records/interface/MuonGeometryRecord.h"
#include "DataFormats/MuonDetId/interface/DTLayerId.h"
#include "DataFormats/DTRecHit/interface/DTRecHit1DPair.h"

#include "RecoLocalMuon/DTRecHit/interface/DTRecHitBaseAlgo.h"
#include "RecoLocalMuon/DTRecHit/interface/DTRecHitAlgoFactory.h"
#include "DataFormats/DTRecHit/interface/DTRecHitCollection.h"

#include "DataFormats/DTDigi/interface/DTDigiCollection.h"



namespace cms
{
class TrackerMuFilter : public edm::EDFilter {
  public:
  TrackerMuFilter(const edm::ParameterSet& conf);
  virtual ~TrackerMuFilter() {}
  //   virtual bool filter(edm::Event & e, edm::EventSetup const& c);
  bool filter(edm::Event & iEvent, edm::EventSetup const& c);

 private:
  edm::ParameterSet conf_;
  bool tracker;
  bool muonDT;
  bool muonDT_MTCC;
  bool muonCSC;
  bool muonRPC;
  std::vector<PSimHit> theStripHits;
  std::vector<PSimHit> theCSCMuonHits;
  std::vector<PSimHit> theDTMuonHits;
  std::vector<PSimHit> theRPCMuonHits;
  };
}

#endif 
