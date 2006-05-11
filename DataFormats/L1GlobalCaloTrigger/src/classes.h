
#include <vector>
#include "DataFormats/L1GlobalCaloTrigger/interface/L1GctCollections.h"
#include "DataFormats/L1GlobalCaloTrigger/interface/L1GctCandidates.h"
#include "DataFormats/L1GlobalCaloTrigger/interface/L1GctEtSums.h"
#include "FWCore/EDProduct/interface/Wrapper.h"

namespace {
  namespace {
    L1GctIsoEmCollection isoEm;
    L1GctNonIsoEmCollection nonIsoEm;
    L1GctCenJetCollection cenJet;
    L1GctForJetCollection forJet;
    L1GctTauJetCollection tauJet;
    L1GctEtTotal etTot;
    L1GctEtHad etHad;
    L1GctEtMiss etMiss;

    edm::Wrapper<L1GctIsoEmCollection> w_isoEm;
    edm::Wrapper<L1GctNonIsoEmCollection> w_nonIsoEm;
    edm::Wrapper<L1GctCenJetCollection> w_cenJet;
    edm::Wrapper<L1GctForJetCollection> w_forJet;
    edm::Wrapper<L1GctTauJetCollection> w_tauJet;
    edm::Wrapper<L1GctEtTotal> w_etTot;
    edm::Wrapper<L1GctEtHad> w_etHad;
    edm::Wrapper<L1GctEtMiss> w_etMiss;
  }
}
