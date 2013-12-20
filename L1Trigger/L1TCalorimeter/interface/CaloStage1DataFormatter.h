///step03
/// \class l1t::CaloStage1DataFormatter
///
/// Description: Convert Stage 2 data formats to Stage 1 data formats
///
/// Implementation:
///
/// \author: Ivan Amos Cali - MIT
///

//

#ifndef CaloStage1DataFormatter_h
#define CaloStage1DataFormatter_h



#include "DataFormats/L1Trigger/interface/BXVector.h"
#include "CondFormats/L1TObjects/interface/FirmwareVersion.h"

#include "DataFormats/L1TCalorimeter/interface/CaloRegion.h"
#include "DataFormats/L1TCalorimeter/interface/CaloEmCand.h"

#include "DataFormats/L1Trigger/interface/EGamma.h"
#include "DataFormats/L1Trigger/interface/Tau.h"
#include "DataFormats/L1Trigger/interface/Jet.h"
#include "DataFormats/L1Trigger/interface/EtSum.h"

#include "DataFormats/L1GlobalCaloTrigger/interface/L1GctCollections.h"

#include <vector>

typedef BXVector<l1t::EGamma> L1TEGammaCollection;
typedef BXVector<l1t::Tau> L1TTauCollection;
typedef BXVector<l1t::Jet> L1TJetCollection;
typedef BXVector<l1t::EtSum> L1TEtSumCollection;


namespace l1t {
    
  class CaloStage1DataFormatter {
  public:
    
    CaloStage1DataFormatter();
    virtual ~CaloStage1DataFormatter();
    
     void ConvertToIsoEmCand(const L1TEGammaCollection &, L1GctEmCandCollection&);
     void ConvertToNonIsoEmCand(L1TEGammaCollection&, L1GctEmCandCollection&);
     void ConvertToCenJetCand(L1TJetCollection&, L1GctJetCandCollection&);
     void ConvertToForJetCand(L1TJetCollection&, L1GctJetCandCollection&);
     void ConvertToTauJetCand(L1TTauCollection&, L1GctJetCandCollection&);
      
     void ConvertToEtTotal(L1TEtSumCollection&, L1GctEtTotalCollection&);
     void ConvertToEtHad(L1TEtSumCollection&,L1GctEtHadCollection&);
     void ConvertToEtMiss(L1TEtSumCollection&,L1GctEtMissCollection&);
     void ConvertToHtMiss(L1TEtSumCollection&,L1GctHtMissCollection&);
     void ConvertToHFBitCounts(L1TEtSumCollection&,L1GctHFBitCountsCollection&);
     void ConvertToHFRingEtSums(L1TEtSumCollection&, L1GctHFRingEtSumsCollection&);
      
     void ConvertToIntJet(L1TJetCollection&, L1GctInternJetDataCollection&);
     void ConvertToIntEtSum(L1TEtSumCollection&,L1GctInternEtSumCollection&);
     void ConvertToIntHtMiss(L1TEtSumCollection&,L1GctInternHtMissCollection&);
    
};
  
} 

#endif
