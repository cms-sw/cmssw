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
HLTCSCActivityFilter::HLTCSCActivityFilter(const edm::ParameterSet& iConfig) : HLTFilter(iConfig),
  m_cscStripDigiTag( iConfig.getParameter<edm::InputTag>("cscStripDigiTag")),
  m_MESR(            iConfig.getParameter<bool>("skipStationRing")),
  m_RingNumb(        iConfig.getParameter<int>("skipRingNumber")),
  m_StationNumb(     iConfig.getParameter<int>("skipStationNumber"))
{
  m_cscStripDigiToken = consumes<CSCStripDigiCollection>(m_cscStripDigiTag);
}

HLTCSCActivityFilter::~HLTCSCActivityFilter() {
}

void
HLTCSCActivityFilter::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  makeHLTFilterDescription(desc);
  desc.add<edm::InputTag>("cscStripDigiTag",edm::InputTag("hltMuonCSCDigis","MuonCSCStripDigi"));
  desc.add<bool>("skipStationRing",true);
  desc.add<int>("skipRingNumber",1);
  desc.add<int>("skipStationNumber",4);
  descriptions.add("hltCSCActivityFilter",desc);
}

//
// member functions
//

// ------------ method called on each new Event  ------------
bool HLTCSCActivityFilter::hltFilter(edm::Event& iEvent, const edm::EventSetup& iSetup, trigger::TriggerFilterObjectWithRefs & filterproduct) const {
  using namespace edm;
  using namespace std;
  using namespace trigger;

  int nStripsFired = 0;

  edm::Handle<CSCStripDigiCollection> cscStrips;
  iEvent.getByToken(m_cscStripDigiToken, cscStrips);

  for (CSCStripDigiCollection::DigiRangeIterator dSDiter=cscStrips->begin(); dSDiter!=cscStrips->end(); ++dSDiter) {
    CSCDetId id = (CSCDetId)(*dSDiter).first;
    bool thisME = ((id.station()== m_StationNumb) && (id.ring()== m_RingNumb));
    if (m_MESR && thisME)
      continue;

    std::vector<CSCStripDigi>::const_iterator stripIter = (*dSDiter).second.first;
    std::vector<CSCStripDigi>::const_iterator lStrip    = (*dSDiter).second.second;
    for( ; stripIter != lStrip; ++stripIter) {
      const std::vector<int> & myADCVals = stripIter->getADCCounts();
      const float pedestal  = 0.5 * (float) (myADCVals[0] + myADCVals[1]);
      const float threshold = 20;
      const float cut = pedestal + threshold;
      for (unsigned int i = 2; i < myADCVals.size(); ++i)
        if (myADCVals[i] > cut) {
          ++nStripsFired;
          break;
        }
    }
  }

  return (nStripsFired >= 1);
}

// define as a framework module
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(HLTCSCActivityFilter);
