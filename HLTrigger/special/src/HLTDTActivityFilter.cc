// -*- C++ -*-
//
// Package:    HLTDTActivityFilter
// Class:      HLTDTActivityFilter
//
/**\class HLTDTActivityFilter HLTDTActivityFilter.cc filter/HLTDTActivityFilter/src/HLTDTActivityFilter.cc

Description:

Implementation:
<Notes on implementation>
*/
//
// Original Author:  Carlo Battilana
//         Created:  Tue Jan 22 13:55:00 CET 2008
// $Id: HLTDTActivityFilter.cc,v 1.4 2009/12/02 12:00:14 gruen Exp $
//
//


// system include files
#include <vector>
#include <string>
#include <map>
#include <iostream>
#include <memory>

// user include files
//#include "DataFormats/Common/interface/Handle.h"
#include "HLTrigger/special/interface/HLTDTActivityFilter.h"
#include "DataFormats/L1DTTrackFinder/interface/L1MuDTChambPhContainer.h"
#include "DataFormats/L1DTTrackFinder/interface/L1MuDTChambThContainer.h"
#include "DataFormats/DTDigi/interface/DTLocalTriggerCollection.h"
#include "DataFormats/DTDigi/interface/DTDigi.h"
#include "DataFormats/DTDigi/interface/DTDigiCollection.h"


//
// constructors and destructor
//
HLTDTActivityFilter::HLTDTActivityFilter(const edm::ParameterSet& iConfig) {

  using namespace std;

  inputDCC_         = iConfig.getParameter<edm::InputTag>("inputDCC");
  inputDDU_         = iConfig.getParameter<edm::InputTag>("inputDDU");
  inputDigis_       = iConfig.getParameter<edm::InputTag>("inputDigis");
  processDCC_       = iConfig.getParameter<bool>("processDCC");
  processDDU_       = iConfig.getParameter<bool>("processDDU");
  processDigis_     = iConfig.getParameter<bool>("processDigis");
  processingMode_   = iConfig.getParameter<int>("processingMode");
  minQual_          = iConfig.getParameter<int>("minQual");
  maxStation_       = iConfig.getParameter<int>("maxStation");
  minChambLayers_   = iConfig.getParameter<int>("minChamberLayers");
  minBX_            = iConfig.getParameter<int>("minDDUBX");
  maxBX_            = iConfig.getParameter<int>("maxDDUBX");
  minActiveChambs_  = iConfig.getParameter<int>("minActiveChambs");

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

// ------------ method called on each new Event  ------------
bool HLTDTActivityFilter::filter(edm::Event& iEvent, const edm::EventSetup& iSetup) {

  using namespace edm;
  using namespace std;

  map<uint32_t,bitset<3> > goodMap;
  if (processDCC_) {

    edm::Handle<L1MuDTChambPhContainer> l1DTTPGPh;
    iEvent.getByLabel(inputDCC_,l1DTTPGPh);
    vector<L1MuDTChambPhDigi>*  phTrigs = l1DTTPGPh->getContainer();
    vector<L1MuDTChambPhDigi>::const_iterator iph  = phTrigs->begin();
    vector<L1MuDTChambPhDigi>::const_iterator iphe = phTrigs->end();
    
    for(; iph !=iphe ; ++iph) {
      int qual = iph->code();
      int ch   = iph->stNum();
      int sec  = iph->scNum() + 1; // DTTF range [0:11] -> DT SC range [1:12] 
      int wh   = iph->whNum();
      if (!activeSecs_[sec]) continue;
      if (ch<=maxStation_ && qual>=minQual_ && qual<7) {
	goodMap[DTChamberId(wh,ch,sec).rawId()].set(0);	
      }
    }
    
  }

  if (processDDU_) {
    
    Handle<DTLocalTriggerCollection> trigsDDU;
    iEvent.getByLabel(inputDDU_,trigsDDU);
    DTLocalTriggerCollection::DigiRangeIterator detUnitIt;
    
    for (detUnitIt=trigsDDU->begin();detUnitIt!=trigsDDU->end();++detUnitIt){
      int ch  = (*detUnitIt).first.station();
      if (!activeSecs_[(*detUnitIt).first.sector()]) continue;
      
      const DTLocalTriggerCollection::Range& range = (*detUnitIt).second;
      for (DTLocalTriggerCollection::const_iterator trigIt = range.first; trigIt!=range.second;++trigIt){	
	int bx = trigIt->bx();
	int qual = trigIt->quality();
	if ( ch<=maxStation_ && bx>=minBX_ && bx<=maxBX_ && qual>=minQual_ && qual<7) {
	  goodMap[(*detUnitIt).first.rawId()].set(1);       
	}
      }
    }

  }

  if (processDigis_) {
    
    edm::Handle<DTDigiCollection> dtdigis;
    iEvent.getByLabel(inputDigis_, dtdigis);
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
	  goodMap[chId.rawId()].set(2);
	}
      }
    }
    
  }
  
  int nGoodCh = 0;
  map<uint32_t,bitset<3> >::const_iterator goodMapIt  = goodMap.begin();
  map<uint32_t,bitset<3> >::const_iterator goodMapEnd = goodMap.end();
  for (; goodMapIt!= goodMapEnd; ++ goodMapIt) {

    bool trigResult = processingMode_%2 ? ((*goodMapIt).second[0] && (*goodMapIt).second[1]) : 
      ((*goodMapIt).second[0] || (*goodMapIt).second[1]);
    bool result     = processingMode_/2 ? (trigResult && (*goodMapIt).second[2]) : 
      (trigResult || (*goodMapIt).second[2]);    
    result && nGoodCh++;

  }
  
  return nGoodCh>=minActiveChambs_;
  
}

// define as a framework module
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(HLTDTActivityFilter);
