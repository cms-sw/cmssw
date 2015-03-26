#ifndef RecoMET_METPUSubtraction_NoPileUpPFMEtProducer_h
#define RecoMET_METPUSubtraction_NoPileUpPFMEtProducer_h

/** \class NoPileUpPFMEtProducer
 *
 * Produce PFMET objects from no-PU jets + "unclustered" no-PU tracks + "unclustered" neutral particles
 * ("unclustered" particles = particles not within jets)
 *
 * \authors Christian Veelken, LLR
 *
 */

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "RecoMET/METPUSubtraction/interface/PFMEtSignInterfaceBase.h"
#include "RecoMET/METPUSubtraction/interface/NoPileUpMEtUtilities.h"

#include "DataFormats/METReco/interface/PFMET.h"
#include "DataFormats/METReco/interface/PFMETFwd.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/Candidate/interface/CandidateFwd.h"


#include <vector>

class NoPileUpPFMEtProducer : public edm::stream::EDProducer<>
{
 public:
  
  NoPileUpPFMEtProducer(const edm::ParameterSet&);
  ~NoPileUpPFMEtProducer();
  
 private:
  
  void produce(edm::Event&, const edm::EventSetup&);
  
  std::string moduleLabel_;

  edm::EDGetTokenT<reco::PFMETCollection> srcMEt_;
  edm::InputTag srcMEtCov_;
  edm::EDGetTokenT<reco::PUSubMETCandInfoCollection> srcJetInfo_;
  edm::EDGetTokenT<reco::PUSubMETCandInfoCollection> srcJetInfoLeptonMatch_;
  edm::EDGetTokenT<reco::PUSubMETCandInfoCollection> srcPFCandInfo_;
  edm::EDGetTokenT<reco::PUSubMETCandInfoCollection> srcPFCandInfoLeptonMatch_;
  typedef std::vector<edm::InputTag>  vInputTag;
  std::vector<edm::EDGetTokenT<reco::CandidateView > > srcLeptons_;

  edm::EDGetTokenT<CorrMETData> srcType0Correction_;

  double sfNoPUjets_;
  double sfNoPUjetOffsetEnCorr_;
  double sfPUjets_;
  double sfNoPUunclChargedCands_;
  double sfPUunclChargedCands_;
  double sfUnclNeutralCands_;
  double sfType0Correction_;
  double sfLeptonIsoCones_;
  
  std::string sfLeptonsName_;
  std::string sfNoPUjetsName_;
  std::string sfNoPUjetOffsetEnCorrName_;
  std::string sfPUjetsName_;
  std::string sfNoPUunclChargedCandsName_;
  std::string sfPUunclChargedCandsName_;
  std::string sfUnclNeutralCandsName_;
  std::string sfType0CorrectionName_;
  std::string sfLeptonIsoConesName_;

  PFMEtSignInterfaceBase* pfMEtSignInterface_;
  double sfMEtCovMin_;
  double sfMEtCovMax_;

  bool saveInputs_;

  int verbosity_;

  NoPileUpMEtUtilities utils_;

};

#endif
