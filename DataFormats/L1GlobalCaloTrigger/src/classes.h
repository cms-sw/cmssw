
#include <vector>
#include "DataFormats/L1GlobalCaloTrigger/interface/L1GctEmCand.h"
#include "DataFormats/L1GlobalCaloTrigger/interface/L1GctJetCand.h"
#include "DataFormats/L1GlobalCaloTrigger/interface/L1GctEtSums.h"
#include "DataFormats/L1GlobalCaloTrigger/interface/L1GctJetCounts.h"
#include "DataFormats/L1GlobalCaloTrigger/interface/L1GctCollections.h"
#include "DataFormats/Common/interface/Wrapper.h"
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/RefProd.h"

namespace {
  namespace {
    L1GctEmCandCollection emCand;
    L1GctJetCandCollection jetCand;
    L1GctEtTotal etTot;
    L1GctEtHad etHad;
    L1GctEtMiss etMiss;
    L1GctJetCounts jetCounts;

    edm::Wrapper<L1GctEmCandCollection> w_emCand;
    edm::Wrapper<L1GctJetCandCollection> w_jetCand;
    edm::Wrapper<L1GctEtTotal> w_etTot;
    edm::Wrapper<L1GctEtHad> w_etHad;
    edm::Wrapper<L1GctEtMiss> w_etMiss;
    edm::Wrapper<L1GctJetCounts> w_jetCounts;

    edm::Ref<L1GctEmCandCollection> emRef ;
    edm::Ref<L1GctJetCandCollection> jetRef ;
    edm::RefProd<L1GctEtTotal> etTotRef ;
    edm::RefProd<L1GctEtHad> etHadRef ;
    edm::RefProd<L1GctEtMiss> etMissRef ;

  }
}
