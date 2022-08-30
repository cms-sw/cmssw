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

// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDFilter.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/L1GlobalMuonTrigger/interface/L1MuRegionalCand.h"
#include "DataFormats/L1GlobalMuonTrigger/interface/L1MuGMTReadoutCollection.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutRecord.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GtPsbWord.h"
//
// class declaration
//

class EcalExclusiveTrigFilter : public edm::one::EDFilter<> {
public:
  explicit EcalExclusiveTrigFilter(const edm::ParameterSet&);
  ~EcalExclusiveTrigFilter() override = default;

private:
  bool filter(edm::Event&, const edm::EventSetup&) override;

  // ----------member data ---------------------------
  const edm::InputTag l1GTReadoutRecTag_;
  const edm::EDGetTokenT<L1MuGMTReadoutCollection> l1GTReadoutRecToken_;
  const edm::EDGetTokenT<L1GlobalTriggerReadoutRecord> l1GTReadoutToken_;
  std::vector<int> l1Accepts_;
  std::vector<std::string> l1Names_;
};
