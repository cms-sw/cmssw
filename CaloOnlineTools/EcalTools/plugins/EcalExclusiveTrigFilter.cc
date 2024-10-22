// -*- C++ -*-
//
// Package:    EcalExclusiveTrigFilter
// Class:      EcalExclusiveTrigFilter
//
/**\class EcalExclusiveTrigFilter EcalExclusiveTrigFilter.cc CaloOnlineTools/EcalExclusiveTrigFilter/src/EcalExclusiveTrigFilter.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Seth COOPER
//         Created:  Thu May 22 11:40:12 CEST 2008
//
//

#include "CaloOnlineTools/EcalTools/plugins/EcalExclusiveTrigFilter.h"

using namespace cms;
using namespace edm;
using namespace std;

//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
EcalExclusiveTrigFilter::EcalExclusiveTrigFilter(const edm::ParameterSet& iConfig)
    : l1GTReadoutRecTag_(iConfig.getUntrackedParameter<std::string>("L1GlobalReadoutRecord", "gtDigis")),
      l1GTReadoutRecToken_(consumes<L1MuGMTReadoutCollection>(l1GTReadoutRecTag_)),
      l1GTReadoutToken_(consumes<L1GlobalTriggerReadoutRecord>(l1GTReadoutRecTag_)) {
  //now do what ever initialization is needed
}

//
// member functions
//

// ------------ method called on each new Event  ------------
bool EcalExclusiveTrigFilter::filter(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  // get the GMTReadoutCollection
  const Handle<L1MuGMTReadoutCollection>& gmtrc_handle = iEvent.getHandle(l1GTReadoutRecToken_);
  L1MuGMTReadoutCollection const* gmtrc = gmtrc_handle.product();
  if (!(gmtrc_handle.isValid())) {
    LogWarning("EcalExclusiveTrigFilter") << "l1MuGMTReadoutCollection"
                                          << " not available";
    //return;
  }

  // get hold of L1GlobalReadoutRecord
  const Handle<L1GlobalTriggerReadoutRecord>& L1GTRR = iEvent.getHandle(l1GTReadoutToken_);
  bool isEcalL1 = false;
  const unsigned int sizeOfDecisionWord(L1GTRR->decisionWord().size());
  if (!(L1GTRR.isValid())) {
    LogWarning("EcalExclusiveTrigFilter") << l1GTReadoutRecTag_ << " not available";
  } else if (sizeOfDecisionWord < 128) {
    LogWarning("EcalExclusiveTrigFilter")
        << "size of L1 decisionword is " << sizeOfDecisionWord << "; L1 Ecal triggering bits not available";
  } else {
    l1Names_.resize(sizeOfDecisionWord);
    l1Accepts_.resize(sizeOfDecisionWord);
    for (unsigned int i = 0; i != sizeOfDecisionWord; ++i) {
      l1Accepts_[i] = 0;
      l1Names_[i] = "NameNotAvailable";
    }
    for (unsigned int i = 0; i != sizeOfDecisionWord; ++i) {
      if (L1GTRR->decisionWord()[i]) {
        l1Accepts_[i]++;
      }
    }

    if (l1Accepts_[14] || l1Accepts_[15] || l1Accepts_[16] || l1Accepts_[17] || l1Accepts_[18] || l1Accepts_[19] ||
        l1Accepts_[20])
      isEcalL1 = true;
    if (l1Accepts_[73] || l1Accepts_[74] || l1Accepts_[75] || l1Accepts_[76] || l1Accepts_[77] || l1Accepts_[78])
      isEcalL1 = true;
  }

  bool isRPCL1 = false;
  bool isDTL1 = false;
  bool isCSCL1 = false;
  bool isHCALL1 = false;

  std::vector<L1MuGMTReadoutRecord> gmt_records = gmtrc->getRecords();
  std::vector<L1MuGMTReadoutRecord>::const_iterator igmtrr;

  for (igmtrr = gmt_records.begin(); igmtrr != gmt_records.end(); igmtrr++) {
    std::vector<L1MuRegionalCand>::const_iterator iter1;
    std::vector<L1MuRegionalCand> rmc;

    //DT triggers
    int idt = 0;
    rmc = igmtrr->getDTBXCands();
    for (iter1 = rmc.begin(); iter1 != rmc.end(); iter1++) {
      if (!(*iter1).empty()) {
        idt++;
      }
    }

    if (igmtrr->getBxInEvent() == 0 && idt > 0)
      isDTL1 = true;

    //RPC triggers
    int irpcb = 0;
    rmc = igmtrr->getBrlRPCCands();
    for (iter1 = rmc.begin(); iter1 != rmc.end(); iter1++) {
      if (!(*iter1).empty()) {
        irpcb++;
      }
    }

    if (igmtrr->getBxInEvent() == 0 && irpcb > 0)
      isRPCL1 = true;

    //CSC Triggers
    int icsc = 0;
    rmc = igmtrr->getCSCCands();
    for (iter1 = rmc.begin(); iter1 != rmc.end(); iter1++) {
      if (!(*iter1).empty()) {
        icsc++;
      }
    }
    if (igmtrr->getBxInEvent() == 0 && icsc > 0)
      isCSCL1 = true;
  }

  L1GlobalTriggerReadoutRecord const* gtrr = L1GTRR.product();

  for (int ibx = -1; ibx <= 1; ibx++) {
    bool hcal_top = false;
    bool hcal_bot = false;
    const L1GtPsbWord psb = gtrr->gtPsbWord(0xbb0d, ibx);
    std::vector<int> valid_phi;
    if ((psb.aData(4) & 0x3f) >= 1) {
      valid_phi.push_back((psb.aData(4) >> 10) & 0x1f);
    }
    if ((psb.bData(4) & 0x3f) >= 1) {
      valid_phi.push_back((psb.bData(4) >> 10) & 0x1f);
    }
    if ((psb.aData(5) & 0x3f) >= 1) {
      valid_phi.push_back((psb.aData(5) >> 10) & 0x1f);
    }
    if ((psb.bData(5) & 0x3f) >= 1) {
      valid_phi.push_back((psb.bData(5) >> 10) & 0x1f);
    }
    std::vector<int>::const_iterator iphi;
    for (iphi = valid_phi.begin(); iphi != valid_phi.end(); iphi++) {
      if (*iphi < 9)
        hcal_top = true;
      if (*iphi > 8)
        hcal_bot = true;
    }
    if (ibx == 0 && hcal_top && hcal_bot)
      isHCALL1 = true;
  }

  edm::LogVerbatim("EcalTools") << "**** Trigger Source ****";
  if (isDTL1)
    edm::LogVerbatim("EcalTools") << "DT";
  if (isRPCL1)
    edm::LogVerbatim("EcalTools") << "RPC";
  if (isCSCL1)
    edm::LogVerbatim("EcalTools") << "CSC";
  if (isHCALL1)
    edm::LogVerbatim("EcalTools") << "HCAL";
  if (isEcalL1)
    edm::LogVerbatim("EcalTools") << "ECAL";
  edm::LogVerbatim("EcalTools") << "************************";

  return (isEcalL1 && !isDTL1 && !isRPCL1 && !isCSCL1 && !isHCALL1);
}
