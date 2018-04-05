#ifndef __L1Analysis_L1AnalysisRecoMet_H__
#define __L1Analysis_L1AnalysisRecoMet_H__

//-------------------------------------------------------------------------------
// Created 03/03/2010 - A.C. Le Bihan
// 
//
// Addition of met reco information
//-------------------------------------------------------------------------------

#include "DataFormats/METReco/interface/CaloMETCollection.h"
#include "DataFormats/METReco/interface/CaloMET.h"
#include "DataFormats/JetReco/interface/CaloJetCollection.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "CondFormats/EcalObjects/interface/EcalChannelStatus.h"
#include "RecoLocalCalo/EcalRecAlgos/interface/EcalSeverityLevelAlgoRcd.h"
#include "RecoLocalCalo/EcalRecAlgos/interface/EcalSeverityLevelAlgo.h"
#include "L1AnalysisRecoMetDataFormat.h"

namespace L1Analysis
{
  class L1AnalysisRecoMet 
  {
  public:
    L1AnalysisRecoMet();
    ~L1AnalysisRecoMet();
    
    void SetMet(const edm::Handle<reco::CaloMETCollection> recoMet);
    void SetHtMht(const edm::Handle<reco::CaloJetCollection> caloJets, float jetptThreshold);
    void SetECALFlags(const edm::ESHandle<EcalChannelStatus> chStatus,
		      const edm::Handle<EcalRecHitCollection> ebRecHits,
		      const edm::Handle<EcalRecHitCollection> eeRecHits,
                      const EcalSeverityLevelAlgo* sevlv);

    L1AnalysisRecoMetDataFormat * getData() {return &recoMet_;}
    void Reset() {recoMet_.Reset();}

  private :
    L1AnalysisRecoMetDataFormat recoMet_;
  }; 
}
#endif


