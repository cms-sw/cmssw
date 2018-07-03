#ifndef __L1Analysis_L1AnalysisRecoTau_H__
#define __L1Analysis_L1AnalysisRecoTau_H__

//-------------------------------------------------------------------------------
// Created 05/03/2010 - A.C. Le Bihan
// 
//
// Original code : L1Trigger/L1TNtuples/L1RecoJetNtupleProducer - Jim Brooke
//-------------------------------------------------------------------------------

#include "JetMETCorrections/JetCorrector/interface/JetCorrector.h"
#include "DataFormats/JetReco/interface/CaloJetCollection.h"
#include "DataFormats/JetReco/interface/JetID.h"
#include "L1AnalysisRecoTauDataFormat.h"

#include "DataFormats/TauReco/interface/PFTau.h"
#include "DataFormats/TauReco/interface/PFTauFwd.h"
#include "DataFormats/TauReco/interface/PFTauDiscriminator.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"


namespace L1Analysis
{
  class L1AnalysisRecoTau
  {
  public:
    L1AnalysisRecoTau();
    ~L1AnalysisRecoTau();
    
    //void Print(std::ostream &os = std::cout) const;
    void SetTau(const edm::Event& event,
					   const edm::EventSetup& setup,
		const edm::Handle<reco::PFTauCollection> taus, const edm::Handle<reco::PFTauDiscriminator> DMFindingOldTaus, const edm::Handle<reco::PFTauDiscriminator> DMFindingTaus, const edm::Handle<reco::PFTauDiscriminator> TightIsoTaus, const edm::Handle<reco::PFTauDiscriminator> TightRawIsoTaus, const edm::Handle<reco::PFTauDiscriminator> LooseIsoTaus, const edm::Handle<reco::PFTauDiscriminator> LooseAntiMuon, const edm::Handle<reco::PFTauDiscriminator> TightAntiMuon, const edm::Handle<reco::PFTauDiscriminator> VLooseAntiElectron, const edm::Handle<reco::PFTauDiscriminator> LooseAntiElectron, const edm::Handle<reco::PFTauDiscriminator> TightAntiElectron, unsigned maxTau);

      /*
(const edm::Event& event,
		    const edm::EventSetup& setup,
		    const edm::Handle<reco::PFTauCollection> taus, 
		    const edm::Handle<reco::PFTauDiscriminator> TightIsoTaus, 
		    const edm::Handle<reco::PFTauDiscriminator> LooseIsoTaus, 
		    //edm::Handle<edm::ValueMap<reco::JetID> > jetsID,
		    //edm::Handle<reco::JetCorrector> jetCorrector,
		    unsigned maxTau);
      */
    L1AnalysisRecoTauDataFormat * getData() {return &recoTau_;}
    void Reset() {recoTau_.Reset();}

  private :
    L1AnalysisRecoTauDataFormat recoTau_;
  }; 
}
#endif


