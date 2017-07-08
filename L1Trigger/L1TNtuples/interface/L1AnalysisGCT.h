#ifndef __L1Analysis_L1AnalysisGCT_H__
#define __L1Analysis_L1AnalysisGCT_H__

//-------------------------------------------------------------------------------
// Created 06/01/2010 - A.C. Le Bihan
// 
// 
// Original code : L1Trigger/L1TNtuples/L1NtupleProducer
//-------------------------------------------------------------------------------

#include "DataFormats/L1GlobalCaloTrigger/interface/L1GctCollections.h"
#include "DataFormats/L1CaloTrigger/interface/L1CaloCollections.h"
#include "DataFormats/Common/interface/Handle.h"

#include "L1AnalysisGCTDataFormat.h"

namespace L1Analysis
{
  class L1AnalysisGCT
  {
  public:
    L1AnalysisGCT();
    L1AnalysisGCT(bool verbose);
    ~L1AnalysisGCT();
    
    void SetJet(edm::Handle < L1GctJetCandCollection > l1CenJets,
                edm::Handle < L1GctJetCandCollection > l1ForJets,
		edm::Handle < L1GctJetCandCollection > l1TauJets,
                edm::Handle < L1GctJetCandCollection > l1IsoTauJets);
		
    void SetES(edm::Handle < L1GctEtMissCollection > l1EtMiss, edm::Handle < L1GctHtMissCollection >  l1HtMiss,
               edm::Handle < L1GctEtHadCollection > l1EtHad, edm::Handle < L1GctEtTotalCollection > l1EtTotal); 	   
    
    void SetHFminbias(edm::Handle < L1GctHFRingEtSumsCollection > l1HFSums, 
                      edm::Handle < L1GctHFBitCountsCollection > l1HFCounts);
		      
    void SetEm(edm::Handle < L1GctEmCandCollection > l1IsoEm, 
               edm::Handle < L1GctEmCandCollection > l1NonIsoEm);

    void Reset() {gct_.Reset();}

    L1AnalysisGCTDataFormat * getData() {return &gct_;}
 
  private :
    bool verbose_;
    L1AnalysisGCTDataFormat gct_;
  }; 
} 
#endif


