
#include <vector>
#include <boost/cstdint.hpp> 
#include "DataFormats/L1GlobalCaloTrigger/interface/L1GctCollections.h"
#include "DataFormats/Common/interface/Wrapper.h"
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/RefProd.h"

namespace {
  namespace {
    // internal collections
    L1GctInternEmCandCollection internEmColl;
    L1GctInternJetDataCollection internJetDataColl;
    L1GctInternEtSumCollection internEtSumColl;
    L1GctFibreCollection fibreWordColl;

    // output collections
    L1GctEmCandCollection emCand;
    L1GctJetCandCollection jetCand;
    L1GctEtTotal etTot;
    L1GctEtHad etHad;
    L1GctEtMiss etMiss;
    L1GctJetCounts jetCounts;
    L1GctEtMissCollection etMissColl;
    L1GctEtTotalCollection etTotColl;
    L1GctEtHadCollection etHadColl;
    L1GctJetCountsCollection jetCountsColl;

    // wrapped internal collections
    edm::Wrapper<L1GctInternEmCandCollection> w_internEmCandColl;
    edm::Wrapper<L1GctInternJetDataCollection> w_internJetDataColl;
    edm::Wrapper<L1GctInternEtSumCollection> w_internEtSumColl;
    edm::Wrapper<L1GctFibreCollection> w_fibreWordColl;

    // wrapped output collections
    edm::Wrapper<L1GctEmCandCollection> w_emCand;
    edm::Wrapper<L1GctJetCandCollection> w_jetCand;
    edm::Wrapper<L1GctEtTotal> w_etTot;
    edm::Wrapper<L1GctEtHad> w_etHad;
    edm::Wrapper<L1GctEtMiss> w_etMiss;
    edm::Wrapper<L1GctJetCounts> w_jetCounts;
    edm::Wrapper<L1GctEtTotalCollection> w_etTotColl;
    edm::Wrapper<L1GctEtHadCollection> w_etHadColl;
    edm::Wrapper<L1GctEtMissCollection> w_etMissColl;
    edm::Wrapper<L1GctJetCountsCollection> w_jetCountsColl;

    // references, used by L1Extra
    edm::Ref<L1GctEmCandCollection> emRef ;
    edm::Ref<L1GctJetCandCollection> jetRef ;
    edm::RefProd<L1GctEtTotal> etTotRef ;
    edm::RefProd<L1GctEtHad> etHadRef ;
    edm::RefProd<L1GctEtMiss> etMissRef ;
    edm::Ref<L1GctEtHadCollection> etHadCollRef ;
    edm::Ref<L1GctEtMissCollection> etMissCollRef ;
    edm::Ref<L1GctEtTotalCollection> etTotCollRef ;

  }
}
