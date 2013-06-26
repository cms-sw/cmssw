// -*- C++ -*-
//
// Package:    HLTDTActivityFilter
// Class:      HLTDTActivityFilter
//


/*

Description: Filter to select events with activity in the muon barrel system

*/


//
// Original Author:  Carlo Battilana
//         Created:  Tue Jan 22 13:55:00 CET 2008
// $Id: HLTDTActivityFilter.cc,v 1.7 2012/01/21 15:00:15 fwyzard Exp $
//
//


#include "HLTrigger/special/interface/HLTDTActivityFilter.h"

// c++ header files
#include <vector>
#include <string>
#include <map>
#include <iostream>
#include <memory>

// Fwk header files
#include "DataFormats/L1GlobalMuonTrigger/interface/L1MuGMTReadoutCollection.h"
#include "DataFormats/L1GlobalMuonTrigger/interface/L1MuRegionalCand.h"
#include "DataFormats/L1DTTrackFinder/interface/L1MuDTChambPhContainer.h"
#include "DataFormats/L1DTTrackFinder/interface/L1MuDTChambThContainer.h"
#include "DataFormats/DTDigi/interface/DTLocalTriggerCollection.h"
#include "DataFormats/DTDigi/interface/DTDigiCollection.h"

#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "Geometry/Records/interface/MuonGeometryRecord.h"
#include "Geometry/DTGeometry/interface/DTGeometry.h"
#include "DataFormats/GeometryVector/interface/Pi.h"


// Typedefs
typedef   std::map<uint32_t,std::bitset<4> > activityMap; // bitset map according to ActivityType enum


//
// constructors and destructor
//
HLTDTActivityFilter::HLTDTActivityFilter(const edm::ParameterSet& iConfig) : HLTFilter(iConfig) {

  using namespace std;

  inputTag_[DCC]    = iConfig.getParameter<edm::InputTag>("inputDCC");
  inputTag_[DDU]    = iConfig.getParameter<edm::InputTag>("inputDDU");
  inputTag_[RPC]    = iConfig.getParameter<edm::InputTag>("inputRPC");
  inputTag_[DIGI]   = iConfig.getParameter<edm::InputTag>("inputDigis");

  process_[DCC]     = iConfig.getParameter<bool>("processDCC");
  process_[DDU]     = iConfig.getParameter<bool>("processDDU");
  process_[RPC]     = iConfig.getParameter<bool>("processRPC");
  process_[DIGI]    = iConfig.getParameter<bool>("processDigis");

  orTPG_            = iConfig.getParameter<bool>("orTPG");
  orRPC_            = iConfig.getParameter<bool>("orRPC");
  orDigi_           = iConfig.getParameter<bool>("orDigi");

  minBX_[DCC]       = iConfig.getParameter<int>("minDCCBX");
  maxBX_[DCC]       = iConfig.getParameter<int>("maxDCCBX");
  minBX_[DDU]       = iConfig.getParameter<int>("minDDUBX");
  maxBX_[DDU]       = iConfig.getParameter<int>("maxDDUBX");
  minBX_[RPC]       = iConfig.getParameter<int>("minRPCBX");
  maxBX_[RPC]       = iConfig.getParameter<int>("maxRPCBX");

  minQual_          = iConfig.getParameter<int>("minTPGQual");
  maxStation_       = iConfig.getParameter<int>("maxStation");
  minChambLayers_   = iConfig.getParameter<int>("minChamberLayers");
  minActiveChambs_  = iConfig.getParameter<int>("minActiveChambs");

  maxDeltaPhi_  = iConfig.getParameter<double>("maxDeltaPhi");
  maxDeltaEta_  = iConfig.getParameter<double>("maxDeltaEta");
  

  activeSecs_.reset();
  vector<int> aSectors = iConfig.getParameter<vector<int> >("activeSectors");
  vector<int>::const_iterator iSec = aSectors.begin();
  vector<int>::const_iterator eSec = aSectors.end();
  for (;iSec!=eSec;++iSec) 
    if ((*iSec)>0 && (*iSec<15)) activeSecs_.set((*iSec)); 

}


HLTDTActivityFilter::~HLTDTActivityFilter() {

}


//
// member functions
//

bool HLTDTActivityFilter::beginRun(edm::Run& iRun, const edm::EventSetup& iSetup) {

  iSetup.get<MuonGeometryRecord>().get(dtGeom_);

  return true;

}

// ------------ method called on each new Event  ------------
bool HLTDTActivityFilter::hltFilter(edm::Event& iEvent, const edm::EventSetup& iSetup, trigger::TriggerFilterObjectWithRefs & filterproduct) {

  using namespace edm;
  using namespace std;

  activityMap actMap;  

  if (process_[DCC]) {

    edm::Handle<L1MuDTChambPhContainer> l1DTTPGPh;
    iEvent.getByLabel(inputTag_[DCC],l1DTTPGPh);
    vector<L1MuDTChambPhDigi>*  phTrigs = l1DTTPGPh->getContainer();
    vector<L1MuDTChambPhDigi>::const_iterator iph  = phTrigs->begin();
    vector<L1MuDTChambPhDigi>::const_iterator iphe = phTrigs->end();
    
    for(; iph !=iphe ; ++iph) {

      int qual = iph->code();
      int bx   = iph->bxNum();
      int ch   = iph->stNum();
      int sec  = iph->scNum() + 1; // DTTF range [0:11] -> DT SC range [1:12] 
      int wh   = iph->whNum();

      if (!activeSecs_[sec]) continue;

      if (ch<=maxStation_ && bx>=minBX_[DCC] && bx<=maxBX_[DCC] 
	  && qual>=minQual_ && qual<7) {
	actMap[DTChamberId(wh,ch,sec).rawId()].set(DCC);	
      }

    }
    
  }

  if (process_[DDU]) {
    
    Handle<DTLocalTriggerCollection> trigsDDU;
    iEvent.getByLabel(inputTag_[DDU],trigsDDU);
    DTLocalTriggerCollection::DigiRangeIterator detUnitIt;
    
    for (detUnitIt=trigsDDU->begin();detUnitIt!=trigsDDU->end();++detUnitIt){

      int ch  = (*detUnitIt).first.station();
      if (!activeSecs_[(*detUnitIt).first.sector()]) continue;
      
      const DTLocalTriggerCollection::Range& range = (*detUnitIt).second;

      for (DTLocalTriggerCollection::const_iterator trigIt = range.first; trigIt!=range.second;++trigIt){	
	int bx = trigIt->bx();
	int qual = trigIt->quality();
	if ( ch<=maxStation_ && bx>=minBX_[DDU] && bx<=maxBX_[DDU] 
	     && qual>=minQual_ && qual<7) {
	  actMap[(*detUnitIt).first.rawId()].set(DDU);       
	}

      }

    }

  }

  if (process_[DIGI]) {
    
    edm::Handle<DTDigiCollection> dtdigis;
    iEvent.getByLabel(inputTag_[DIGI], dtdigis);
    std::map<uint32_t,int> hitMap;
    DTDigiCollection::DigiRangeIterator dtLayerIdIt;
    
    for (dtLayerIdIt=dtdigis->begin(); dtLayerIdIt!=dtdigis->end(); dtLayerIdIt++) {

      DTChamberId chId = ((*dtLayerIdIt).first).chamberId();
      if (!activeSecs_[(*dtLayerIdIt).first.sector()]) continue;
      uint32_t rawId = chId.rawId();
      int station = chId.station();

      if (station<=maxStation_) {
	if (hitMap.find(rawId)!=hitMap.end()) {
	  hitMap[rawId]++;
	} else {
	  hitMap[rawId]=1;
	}
	if (hitMap[rawId]>=minChambLayers_) {
	  actMap[chId.rawId()].set(DIGI);
	}
      }

    }
    
  }

  if (process_[RPC]) {

    edm::Handle<L1MuGMTReadoutCollection> gmtrc; 
    iEvent.getByLabel(inputTag_[RPC],gmtrc);

    std::vector<L1MuGMTReadoutRecord> gmtrr = gmtrc->getRecords();
    std::vector<L1MuGMTReadoutRecord>::const_iterator recIt  = gmtrr.begin();
    std::vector<L1MuGMTReadoutRecord>::const_iterator recEnd = gmtrr.end();

    for(; recIt!=recEnd; ++recIt) {

      std::vector<L1MuRegionalCand> rpcCands = (*recIt).getBrlRPCCands();
      std::vector<L1MuRegionalCand>::const_iterator candIt  = rpcCands.begin();
      std::vector<L1MuRegionalCand>::const_iterator candEnd = rpcCands.end();
      
      for(; candIt!=candEnd; ++candIt) {
	
	if (candIt->empty()) continue;
	int bx = (*candIt).bx();

	if (bx>=minBX_[RPC] && bx<=maxBX_[RPC]) {
	  activityMap::iterator actMapIt  = actMap.begin();
	  activityMap::iterator actMapEnd = actMap.end();
	  for (; actMapIt!= actMapEnd; ++ actMapIt)
	    if (matchChamber((*actMapIt).first,(*candIt))) 
	      (*actMapIt).second.set(RPC);
	}
      }
    }

  }
  
  int nActCh = 0;
  activityMap::const_iterator actMapIt  = actMap.begin();
  activityMap::const_iterator actMapEnd = actMap.end();

  for (; actMapIt!=actMapEnd; ++actMapIt) 
    hasActivity((*actMapIt).second) && nActCh++ ;

  bool result = nActCh>=minActiveChambs_;

  return result;
  
}


bool HLTDTActivityFilter::hasActivity(const std::bitset<4>& actWord) {

  bool actTPG   = orTPG_   ? actWord[DCC] || actWord[DDU] : actWord[DCC] && actWord[DDU];
  bool actTrig  = orRPC_   ? actWord[RPC] || actTPG : actWord[RPC] && actTPG;
  bool result   = orDigi_  ? actWord[DIGI] || actTrig : actWord[DIGI] && actTrig; 

  return result;

}

bool HLTDTActivityFilter::matchChamber(const uint32_t& rawId, const L1MuRegionalCand& rpcTrig) {

  const GlobalPoint chPos = dtGeom_->chamber(DTChamberId(rawId))->position();
  
  float fDeltaPhi = fabs( chPos.phi() - rpcTrig.phiValue() );
  if ( fDeltaPhi>Geom::pi() ) fDeltaPhi = fabs(fDeltaPhi - 2*Geom::pi());

  float fDeltaEta = fabs( chPos.eta() - rpcTrig.etaValue() );

  bool result = fDeltaPhi<maxDeltaPhi_ && fDeltaEta<maxDeltaEta_;

  return result;

}


// define as a framework module
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(HLTDTActivityFilter);
