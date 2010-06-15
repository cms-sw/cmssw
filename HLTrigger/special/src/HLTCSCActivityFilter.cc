// -*- C++ -*-
//
// Package:    HLTCSCActivityFilter
// Class:      HLTCSCActivityFilter
//
/**\class HLTCSCActivityFilter HLTCSCActivityFilter.cc filter/HLTCSCActivityFilter/src/HLTCSCActivityFilter.cc

Description:

Implementation:
<Notes on implementation>
*/
//
// Original Author:  Carlo Battilana
//         Created:  Tue Jan 22 13:55:00 CET 2008
// $Id: HLTCSCActivityFilter.cc,v 1.2 2010/06/15 16:09:57 fwyzard Exp $
//
//

// system include files
#include <vector>
#include <map>
#include <iostream>
#include <memory>

// user include files
#include "HLTrigger/special/interface/HLTCSCActivityFilter.h"

//
// constructors and destructor
//
HLTCSCActivityFilter::HLTCSCActivityFilter(const edm::ParameterSet& iConfig) {
  using namespace std;

  m_applyfilter     = iConfig.getParameter<bool>("applyfilter");
  m_cscStripDigiTag = iConfig.getParameter<edm::InputTag>("cscStripDigiTag");
  m_processDigis    = iConfig.getParameter<bool>("processDigis");
  m_MESR            = iConfig.getParameter<bool>("StationRing");  
  m_StationNumb     = iConfig.getParameter<int>("StationNumber");
  m_RingNumb        = iConfig.getParameter<int>("RingNumber");
}

HLTCSCActivityFilter::~HLTCSCActivityFilter() {
}

//
// member functions
//

// ------------ method called on each new Event  ------------
bool HLTCSCActivityFilter::filter(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  using namespace edm;
  using namespace std;

  bool accepted = false;
  int nStripsFired = 0;

  if (m_processDigis) {    
    edm::Handle<CSCStripDigiCollection> cscStrips;
    iEvent.getByLabel(m_cscStripDigiTag,cscStrips);
    
    for (CSCStripDigiCollection::DigiRangeIterator dSDiter=cscStrips->begin(); dSDiter!=cscStrips->end(); dSDiter++) {
      CSCDetId id = (CSCDetId)(*dSDiter).first;
      bool thisME = ((id.station()== m_StationNumb) && (id.ring()== m_RingNumb));
      if (m_MESR && thisME)continue;

      std::vector<CSCStripDigi>::const_iterator stripIter = (*dSDiter).second.first;
      std::vector<CSCStripDigi>::const_iterator lStrip    = (*dSDiter).second.second;
      for( ; stripIter != lStrip; ++stripIter) {
	std::vector<int> myADCVals = stripIter->getADCCounts();
	bool thisStripFired = false;
	float thisPedestal = 0.5*(float)(myADCVals[0]+myADCVals[1]);
	float diff = 0.;
	float threshold = 20;
	for (unsigned int iCount = 0; iCount < myADCVals.size(); iCount++) {
	  diff = (float)myADCVals[iCount]-thisPedestal;
	  if (diff > threshold) { thisStripFired = true; }
	} 
	if (thisStripFired) {
	  nStripsFired++;
	}
      }
    }
  }    
    bool b_Strips = false;
    if(nStripsFired >= 1) b_Strips = true;
    if(b_Strips)accepted = true;
    ////////////////////////////////////// DONE //////////////////////////////
    if (m_applyfilter)
      return accepted;
    else
      return true; 
} 

// define as a framework module
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(HLTCSCActivityFilter);
