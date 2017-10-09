#ifndef DataFormats_TauReco_PFTauTagInfo_h
#define DataFormats_TauReco_PFTauTagInfo_h

/* class PFTauTagInfo
 * the object of this class is created by RecoTauTag/RecoTau PFRecoTauTagInfoProducer EDProducer starting from JetTrackAssociations <a PFJet,a list of Tracks> object
 *                          is the initial object for building a PFTau object
 * created: Aug 29 2007,
 * revised: Sep 10 2007,
 * authors: Ludovic Houchu
 */

#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidateFwd.h"
#include "DataFormats/JetReco/interface/PFJetCollection.h"
#include "DataFormats/TauReco/interface/PFTauTagInfoFwd.h"
#include "DataFormats/TauReco/interface/BaseTauTagInfo.h"


namespace reco{ 
  class PFTauTagInfo : public BaseTauTagInfo {
  public:
    PFTauTagInfo(){}
    virtual ~PFTauTagInfo(){};
    virtual PFTauTagInfo* clone()const;
    
    //get the PFCandidates which compose the PF jet and were filtered by RecoTauTag/TauTagTools/ TauTagTools::filteredPFChargedHadrCands(.,...), filteredPFNeutrHadrCands(.), filteredPFGammaCands(.) functions through RecoTauTag/RecoTauTag/ PFRecoTauTagInfoProducer EDProducer
    std::vector<reco::PFCandidatePtr> PFCands()const;
    const std::vector<reco::PFCandidatePtr>& PFChargedHadrCands()const;
    void  setPFChargedHadrCands(const std::vector<reco::PFCandidatePtr>&);
    const std::vector<reco::PFCandidatePtr>& PFNeutrHadrCands()const;
    void  setPFNeutrHadrCands(const std::vector<reco::PFCandidatePtr>&);
    const std::vector<reco::PFCandidatePtr>& PFGammaCands()const;
    void  setPFGammaCands(const std::vector<reco::PFCandidatePtr>&);
    
    //the reference to the PFJet
    const PFJetRef& pfjetRef()const;
    void setpfjetRef(const PFJetRef);
  private:
    PFJetRef PFJetRef_;
    std::vector<reco::PFCandidatePtr> PFChargedHadrCands_;
    std::vector<reco::PFCandidatePtr> PFNeutrHadrCands_;
    std::vector<reco::PFCandidatePtr> PFGammaCands_;
  };
}
#endif

