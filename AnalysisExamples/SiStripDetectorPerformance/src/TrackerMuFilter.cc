//massimiliano.chiorboli@cern.ch

#include "EventFilter/SiStripChannelChargeFilter/interface/MTCCHLTrigger.h"
#include "FWCore/Framework/interface/Handle.h"
#include "DataFormats/SiStripCluster/interface/SiStripCluster.h"
#include "DataFormats/SiStripDigi/interface/SiStripDigi.h"
#include "DataFormats/Common/interface/DetSetVector.h"
#include "AnalysisExamples/SiStripDetectorPerformance/interface/TrackerMuFilter.h"

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
TrackerMuFilter::TrackerMuFilter(const edm::ParameterSet& conf):    conf_(conf)
{
  tracker=conf_.getUntrackedParameter<bool>("TrackerHits");
  muonDT=conf_.getUntrackedParameter<bool>("DTMuonHits");
  muonCSC=conf_.getUntrackedParameter<bool>("CSCMuonHits");
  muonRPC=conf_.getUntrackedParameter<bool>("RPCMuonHits");
  //theDTDigiLabel = conf_.getParameter<string>("dtDigiLabel");
}

bool TrackerMuFilter::filter(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  //  Step A: Get Inputs
  theStripHits.clear();
  edm::Handle<edm::PSimHitContainer> TIBHitsLowTof;
  edm::Handle<edm::PSimHitContainer> TIBHitsHighTof;
  edm::Handle<edm::PSimHitContainer> TOBHitsLowTof;
  edm::Handle<edm::PSimHitContainer> TOBHitsHighTof;
  edm::Handle<edm::PSimHitContainer> TECHitsLowTof;
  edm::Handle<edm::PSimHitContainer> TECHitsHighTof;

  iEvent.getByLabel("g4SimHits","TrackerHitsTIBLowTof", TIBHitsLowTof);
  iEvent.getByLabel("g4SimHits","TrackerHitsTIBHighTof", TIBHitsHighTof);
  iEvent.getByLabel("g4SimHits","TrackerHitsTOBLowTof", TOBHitsLowTof);
  iEvent.getByLabel("g4SimHits","TrackerHitsTOBHighTof", TOBHitsHighTof);
  iEvent.getByLabel("g4SimHits","TrackerHitsTECLowTof", TECHitsLowTof);
  iEvent.getByLabel("g4SimHits","TrackerHitsTECHighTof", TECHitsHighTof);
    
  theStripHits.insert(theStripHits.end(), TIBHitsLowTof->begin(), TIBHitsLowTof->end());
  theStripHits.insert(theStripHits.end(), TIBHitsHighTof->begin(), TIBHitsHighTof->end());
  theStripHits.insert(theStripHits.end(), TOBHitsLowTof->begin(), TOBHitsLowTof->end()); 
  theStripHits.insert(theStripHits.end(), TOBHitsHighTof->begin(), TOBHitsHighTof->end());
  theStripHits.insert(theStripHits.end(), TECHitsLowTof->begin(), TECHitsLowTof->end()); 
  theStripHits.insert(theStripHits.end(), TECHitsHighTof->begin(), TECHitsHighTof->end());

  // Get the DT Geometry
  edm::ESHandle<DTGeometry> dtGeom;
  iSetup.get<MuonGeometryRecord>().get(dtGeom);

//   // Get the digis from the event
//   Handle<DTDigiCollection> digis;
//   iEvent.getByLabel("muonDTDigis", digis);



  theCSCMuonHits.clear();
  edm::Handle<edm::PSimHitContainer> muCSCHits;
  iEvent.getByLabel("g4SimHits","MuonCSCHits",muCSCHits);

  theCSCMuonHits.insert(theCSCMuonHits.end(), muCSCHits->begin(), muCSCHits->end());



  theDTMuonHits.clear();
  muonDT_MTCC = false;
  edm::Handle<edm::PSimHitContainer> muDTHits; 
  iEvent.getByLabel("g4SimHits","MuonDTHits",muDTHits);
  theDTMuonHits.insert(theDTMuonHits.end(), muDTHits->begin(), muDTHits->end());
  edm::PSimHitContainer::const_iterator iDTSimHit;
  for (iDTSimHit = muDTHits->begin(); iDTSimHit != muDTHits->end(); iDTSimHit++) {
    //    std::cout << "muDTHits->DetUnit = " << (*iDTSimHit).detUnitId() << std::endl;
    DTWireId wireIdSim( (*iDTSimHit).detUnitId() );
    const DTLayer* hittedLayer = dtGeom->layer(wireIdSim);
    const DTLayerId& layerId = hittedLayer->id();
    //    std::cout << "DTLayerId->layer() = " << layerId.layer() << std::endl;
    DTSuperLayerId SLId = layerId.superlayerId(); //need slId
    //    std::cout <<"SLId.wheel() = " << SLId.wheel() << " SLId.sector() = " << SLId.sector() << " SLId.station() = " << SLId.station() << " SLId.superlayer() = " << SLId.superlayer() << std::endl;

//     //CHECK IF LAYER IN READ OUT IN MTCC SETUP !!!!!!!!!!!!!!!!!!!
    if ( (SLId.wheel()==1 && SLId.sector()==10 && SLId.station()==1 && SLId.superlayer()==1) || 
	 (SLId.wheel()==1 && SLId.sector()==10 && SLId.station()==1 && SLId.superlayer()==2) || 
	 (SLId.wheel()==1 && SLId.sector()==10 && SLId.station()==1 && SLId.superlayer()==3) || 
	 (SLId.wheel()==1 && SLId.sector()==10 && SLId.station()==2 && SLId.superlayer()==1) || 
	 (SLId.wheel()==1 && SLId.sector()==10 && SLId.station()==2 && SLId.superlayer()==2) || 
	 (SLId.wheel()==1 && SLId.sector()==10 && SLId.station()==2 && SLId.superlayer()==3) || 
	 (SLId.wheel()==1 && SLId.sector()==10 && SLId.station()==3 && SLId.superlayer()==1) || 
	 (SLId.wheel()==1 && SLId.sector()==10 && SLId.station()==3 && SLId.superlayer()==2) || 
	 (SLId.wheel()==1 && SLId.sector()==10 && SLId.station()==3 && SLId.superlayer()==3) || 
	 (SLId.wheel()==1 && SLId.sector()==10 && SLId.station()==4 && SLId.superlayer()==1) || 
	 (SLId.wheel()==1 && SLId.sector()==10 && SLId.station()==4 && SLId.superlayer()==3) || 
	 (SLId.wheel()==1 && SLId.sector()==14 && SLId.station()==4 && SLId.superlayer()==1) || 
	 (SLId.wheel()==1 && SLId.sector()==14 && SLId.station()==4 && SLId.superlayer()==3) || 
	 (SLId.wheel()==2 && SLId.sector()==10 && SLId.station()==1 && SLId.superlayer()==1) || 
	 (SLId.wheel()==2 && SLId.sector()==10 && SLId.station()==1 && SLId.superlayer()==2) || 
	 (SLId.wheel()==2 && SLId.sector()==10 && SLId.station()==1 && SLId.superlayer()==3) || 
	 (SLId.wheel()==2 && SLId.sector()==10 && SLId.station()==2 && SLId.superlayer()==1) || 
	 (SLId.wheel()==2 && SLId.sector()==10 && SLId.station()==2 && SLId.superlayer()==2) || 
	 (SLId.wheel()==2 && SLId.sector()==10 && SLId.station()==2 && SLId.superlayer()==3) ||
	 (SLId.wheel()==2 && SLId.sector()==10 && SLId.station()==3 && SLId.superlayer()==1) ||
	 (SLId.wheel()==2 && SLId.sector()==10 && SLId.station()==3 && SLId.superlayer()==2) || 
	 (SLId.wheel()==2 && SLId.sector()==10 && SLId.station()==3 && SLId.superlayer()==3) ||
	 (SLId.wheel()==2 && SLId.sector()==10 && SLId.station()==4 && SLId.superlayer()==1) ||
	 (SLId.wheel()==2 && SLId.sector()==10 && SLId.station()==4 && SLId.superlayer()==3) ||
	 (SLId.wheel()==2 && SLId.sector()==14 && SLId.station()==4 && SLId.superlayer()==1) ||
	 (SLId.wheel()==2 && SLId.sector()==14 && SLId.station()==4 && SLId.superlayer()==3) ||
	 (SLId.wheel()==2 && SLId.sector()==11 && SLId.station()==1 && SLId.superlayer()==1) ||
	 (SLId.wheel()==2 && SLId.sector()==11 && SLId.station()==1 && SLId.superlayer()==2) || 
	 (SLId.wheel()==2 && SLId.sector()==11 && SLId.station()==1 && SLId.superlayer()==3) ||
	 (SLId.wheel()==2 && SLId.sector()==11 && SLId.station()==2 && SLId.superlayer()==1) ||
	 (SLId.wheel()==2 && SLId.sector()==11 && SLId.station()==2 && SLId.superlayer()==2) || 
	 (SLId.wheel()==2 && SLId.sector()==11 && SLId.station()==2 && SLId.superlayer()==3) ||
	 (SLId.wheel()==2 && SLId.sector()==11 && SLId.station()==3 && SLId.superlayer()==1) ||
	 (SLId.wheel()==2 && SLId.sector()==11 && SLId.station()==3 && SLId.superlayer()==2) || 
	 (SLId.wheel()==2 && SLId.sector()==11 && SLId.station()==3 && SLId.superlayer()==3) ||
	 (SLId.wheel()==2 && SLId.sector()==11 && SLId.station()==4 && SLId.superlayer()==1) ||
	 (SLId.wheel()==2 && SLId.sector()==11 && SLId.station()==4 && SLId.superlayer()==3) ) muonDT_MTCC = true;
  }

  theRPCMuonHits.clear();
  edm::Handle<edm::PSimHitContainer> muRPCHits; 
  iEvent.getByLabel("g4SimHits","MuonRPCHits",muRPCHits);

  theRPCMuonHits.insert(theRPCMuonHits.end(), muRPCHits->begin(), muRPCHits->end()); 

  if(tracker&&muonCSC){
    if(theStripHits.size()&&theCSCMuonHits.size()){
      return true;
    }else{
      return false;
    }
  }

  std::cout << "*************************** theStripHits.size()   = " << theStripHits.size()   << std::endl;
  std::cout << "*************************** theDTMuonHits.size()  = " << theDTMuonHits.size()  << std::endl;
  std::cout << "*************************** muonDT_MTCC           = " << (bool)muonDT_MTCC           << std::endl;
  std::cout << "*************************** theRPCMuonHits.size() = " << theRPCMuonHits.size() << std::endl;
  std::cout << "*************************** theCSCMuonHits.size() = " << theCSCMuonHits.size() << std::endl;

  if(tracker&&muonDT&&muonDT_MTCC){
    if(theStripHits.size()&&theDTMuonHits.size()){
          std::cout << "*************************************************** Tracker & MuonDT ***********************************" << std::endl;
      return true;
    }else{
          std::cout << "*************************************************** NOT Tracker & MuonDT ***********************************" << std::endl;
      return false;
    }
  }

  if(tracker&&muonRPC){
    if(theStripHits.size()&&theRPCMuonHits.size()){
          std::cout << "*************************************************** Tracker & MuonRPC ***********************************" << std::endl;
      return true;
    }else{
          std::cout << "*************************************************** NOT Tracker & MuonRPC ***********************************" << std::endl;
      return false;
    }
  }

  if(tracker){
    if(theStripHits.size()){
          std::cout << "*************************************************** Tracker ***********************************" << std::endl;
      return true;
    }else{
          std::cout << "*************************************************** NOT Tracker ***********************************" << std::endl;
      return false;
    }
  }

  if(muonRPC){
    if(theRPCMuonHits.size()){
           std::cout << "*************************************************** MuonRPC ***********************************" << std::endl;
      return true;
    }else{
           std::cout << "*************************************************** NOT MuonRPC ***********************************" << std::endl;
      return false;
    }
  }

  if(muonDT&&muonDT_MTCC){
    if(theDTMuonHits.size()){
           std::cout << "*************************************************** MuonDT ***********************************" << std::endl;
      return true;
    }else{
           std::cout << "*************************************************** NOT MuonDT ***********************************" << std::endl;
      return false;
    }
  }

  if(muonCSC){
    if(theCSCMuonHits.size()){
           std::cout << "*************************************************** MuonCSC ***********************************" << std::endl;
      return true;
    }else{
           std::cout << "*************************************************** NOT MuonCSC ***********************************" << std::endl;
      return false;
    }
  }

  return true;
}

}
