#ifndef RecoTauTag_TauAnalysisTools_AntiElectronDiscrMVATrainingNtupleProducer_h
#define RecoTauTag_TauAnalysisTools_AntiElectronDiscrMVATrainingNtupleProducer_h

#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidateFwd.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidateElectronExtra.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidateElectronExtraFwd.h"
#include "DataFormats/TauReco/interface/PFTau.h"
#include "DataFormats/TauReco/interface/PFTauFwd.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrackReco/interface/HitPattern.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrack.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrackFwd.h"
#include "DataFormats/CaloRecHit/interface/CaloCluster.h"
#include "DataFormats/CaloRecHit/interface/CaloClusterFwd.h"
#include "DataFormats/EgammaReco/interface/SuperCluster.h"
#include "DataFormats/EgammaReco/interface/SuperClusterFwd.h"
#include "DataFormats/Math/interface/deltaR.h"

#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include <TFile.h>
#include <TTree.h>
#include <TMath.h>

#include <vector>
#include <string>

class AntiElectronDiscrMVATrainingNtupleProducer : public edm::EDAnalyzer
{
 public:
  // constructor 
  explicit AntiElectronDiscrMVATrainingNtupleProducer(const edm::ParameterSet&);
    
  // destructor
  ~AntiElectronDiscrMVATrainingNtupleProducer();
  
 private:
  void beginJob();
  void analyze(const edm::Event&, const edm::EventSetup&);
  void endJob();
  
  edm::InputTag srcPFTaus_;
  edm::InputTag srcGsfElectrons_;
  edm::InputTag srcPrimaryVertex_;
  edm::InputTag srcGenElectrons_;
  edm::InputTag srcGenTaus_;

  struct tauIdDiscrEntryType
  {
    tauIdDiscrEntryType(const std::string& name, const edm::InputTag& src)
      : src_(src)
    {
      branchName_ = name;
    }
    ~tauIdDiscrEntryType() {}
    edm::InputTag src_;
    std::string branchName_;
    float value_;
  };
  std::vector<tauIdDiscrEntryType> tauIdDiscrEntries_;

  typedef std::vector<edm::InputTag> vInputTag;
  vInputTag srcWeights_;
  
  int verbosity_;
  
  TTree* tree_;

  unsigned long run_;
  unsigned long event_;
  unsigned long lumi_;
  int NumPV_;
  int NumGsfEle_;
  int NumPFTaus_;
  int NumPatTaus_;
  int NumGenEle_;
  int NumGenHad_;
  int NumGenJet_;

  std::vector<float> GammasdEta_;
  std::vector<float> GammasdPhi_;
  std::vector<float> GammasPt_;
  int Tau_GsfEleMatch_;
  int Tau_GenEleMatch_;
  int Tau_GenHadMatch_;
  float Tau_Eta_;
  float Tau_EtaAtEcalEntrance_;
  float Tau_PhiAtEcalEntrance_;
  float Tau_EtaAtEcalEntranceEcalEnWeighted_;
  float Tau_PhiAtEcalEntranceEcalEnWeighted_;
  float Tau_LeadNeutralPFCandEtaAtEcalEntrance_;
  float Tau_LeadNeutralPFCandPhiAtEcalEntrance_;
  float Tau_LeadNeutralPFCandPt_;
  float Tau_LeadChargedPFCandEtaAtEcalEntrance_;
  float Tau_LeadChargedPFCandPhiAtEcalEntrance_;
  float Tau_LeadChargedPFCandPt_;
  float Tau_Pt_;
  float Tau_LeadHadronPt_;
  float Tau_Phi_;
  int Tau_HasGsf_; 
  float Tau_GSFChi2_; 
  int Tau_GSFNumHits_; 
  int Tau_GSFNumPixelHits_; 
  int Tau_GSFNumStripHits_; 
  float Tau_GSFTrackResol_; 
  float Tau_GSFTracklnPt_; 
  float Tau_GSFTrackEta_; 
  int Tau_HasKF_; 
  float Tau_KFChi2_; 
  int Tau_KFNumHits_; 
  int Tau_KFNumPixelHits_; 
  int Tau_KFNumStripHits_; 
  float Tau_KFTrackResol_; 
  float Tau_KFTracklnPt_; 
  float Tau_KFTrackEta_; 
  float Tau_EmFraction_; 
  int Tau_NumChargedCands_;
  int Tau_NumGammaCands_; 
  float Tau_HadrHoP_; 
  float Tau_HadrEoP_; 
  float Tau_VisMass_; 
  float Tau_GammaEtaMom_;
  float Tau_GammaPhiMom_;
  float Tau_GammaEnFrac_;
  float Tau_HadrMva_; 
  int Tau_DecayMode_;
  int Tau_MatchElePassVeto_;
  float Tau_VtxZ_;
  float Tau_zImpact_;

  int Elec_GenEleMatch_;
  int Elec_GenEleFromZMatch_;
  int Elec_GenEleFromZTauTauMatch_;
  int Elec_GenHadMatch_;
  int Elec_GenJetMatch_;
  float Elec_AbsEta_;
  float Elec_Pt_;
  int Elec_HasSC_;
  float Elec_PFMvaOutput_;
  float Elec_Ee_;
  float Elec_Egamma_;
  float Elec_Pin_;
  float Elec_Pout_;
  float Elec_EtotOverPin_;
  float Elec_EeOverPout_;
  float Elec_EgammaOverPdif_;
  int Elec_EarlyBrem_;
  int Elec_LateBrem_;
  float Elec_Logsihih_;
  float Elec_DeltaEta_;
  float Elec_HoHplusE_;
  float Elec_Fbrem_;
  int Elec_HasKF_;
  float Elec_Chi2KF_;
  int Elec_KFNumHits_;
  int Elec_KFNumPixelHits_;
  int Elec_KFNumStripHits_;
  float Elec_KFTrackResol_;
  float Elec_KFTracklnPt_;
  float Elec_KFTrackEta_;
  int Elec_HasGSF_;
  float Elec_Chi2GSF_;
  int Elec_GSFNumHits_;
  int Elec_GSFNumPixelHits_;
  int Elec_GSFNumStripHits_;
  float Elec_GSFTrackResol_;
  float Elec_GSFTracklnPt_;
  float Elec_GSFTrackEta_;

  int ElecVeto_N_;
  float ElecVeto_Pt_;
  float ElecVeto_Eta_;
  float ElecVeto_Phi_;

  float evtWeight_;
};

#endif   
