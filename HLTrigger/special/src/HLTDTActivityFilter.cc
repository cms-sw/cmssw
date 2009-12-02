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
// $Id: HLTDTActivityFilter.cc,v 1.2 2009/11/26 17:02:58 fwyzard Exp $
//
//


// system include files
#include <vector>
#include <string>
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

  bool goodDCC(false);
  if (processDCC_) {

    edm::Handle<L1MuDTChambPhContainer> l1DTTPGPh;
    iEvent.getByLabel(inputDCC_,l1DTTPGPh);
    vector<L1MuDTChambPhDigi>*  phTrigs = l1DTTPGPh->getContainer();
    vector<L1MuDTChambPhDigi>::const_iterator iph  = phTrigs->begin();
    vector<L1MuDTChambPhDigi>::const_iterator iphe = phTrigs->end();

    int nPrimitives=0;
    for(; iph !=iphe ; ++iph) {
      int qual = iph->code();
      nPrimitives += (iph->stNum()<=maxStation_ && qual>=minQual_ && qual<7);
      if (nPrimitives>=minActiveChambs_) { goodDCC=true; break; }
    }
  }

  bool goodDDU(false);
  if (processDDU_) {

    Handle<DTLocalTriggerCollection> trigsDDU;
    iEvent.getByLabel(inputDDU_,trigsDDU);
    DTLocalTriggerCollection::DigiRangeIterator detUnitIt;
    int nPrimitives=0;

    for (detUnitIt=trigsDDU->begin();detUnitIt!=trigsDDU->end();++detUnitIt){
      int chamb = (*detUnitIt).first.station();
      const DTLocalTriggerCollection::Range& range = (*detUnitIt).second;
      for (DTLocalTriggerCollection::const_iterator trigIt = range.first; trigIt!=range.second;++trigIt){	
	int bx = trigIt->bx();
	int qual = trigIt->quality();
	nPrimitives += ( chamb<=maxStation_ && bx>=minBX_ && bx<=maxBX_ && qual>=minQual_ && qual<7);
	if (nPrimitives>=minActiveChambs_) { goodDDU=true; break; }
      }
      if (goodDDU) { break; }
    }
  }

  bool goodDigis(false);
  if (processDigis_) {

    edm::Handle<DTDigiCollection> dtdigis;
    iEvent.getByLabel(inputDigis_, dtdigis);
    std::map<uint32_t,int> hitMap;
    int activeChambDigis = 0;
    DTDigiCollection::DigiRangeIterator dtLayerIdIt;

    for (dtLayerIdIt=dtdigis->begin(); dtLayerIdIt!=dtdigis->end(); dtLayerIdIt++) {
      DTChamberId chId = ((*dtLayerIdIt).first).chamberId();
      uint32_t rawId = chId.rawId();
      int station = chId.station();
      if (station<=maxStation_) {
	if (hitMap.find(rawId)!=hitMap.end()) {
	  hitMap[rawId]++;
	} else {
	  hitMap[rawId]=1;
	}
	if (hitMap[rawId]>=minChambLayers_) {
	  activeChambDigis++;
	  if (activeChambDigis>=minActiveChambs_) {
	    goodDigis = true;
	    break;
	  }
	}
      }
    }

  }


  bool trigResult = processingMode_%2 ? (goodDCC && goodDDU) : (goodDCC || goodDDU);
  bool result     = processingMode_/2 ? (trigResult && goodDigis) : (trigResult || goodDigis);

  return result;

}
