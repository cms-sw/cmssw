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
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "RecoMET/METPUSubtraction/interface/PFMEtSignInterfaceBase.h"
#include "RecoMET/METPUSubtraction/interface/noPileUpMEtUtilities.h"

#include "DataFormats/METReco/interface/PFMET.h"
#include "DataFormats/METReco/interface/PFMETFwd.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "DataFormats/Candidate/interface/Candidate.h"


#include <vector>

class NoPileUpPFMEtProducer : public edm::EDProducer
{
 public:
  
  NoPileUpPFMEtProducer(const edm::ParameterSet&);
  ~NoPileUpPFMEtProducer();
  
 private:
  
  void produce(edm::Event&, const edm::EventSetup&);
  
  std::string moduleLabel_;

  edm::EDGetTokenT<reco::PFMETCollection> srcMEt_;
  edm::InputTag srcMEtCov_;
  edm::EDGetTokenT<reco::MVAMEtJetInfoCollection> srcJetInfo_;
  edm::EDGetTokenT<reco::MVAMEtJetInfoCollection> srcJetInfoLeptonMatch_;
  edm::EDGetTokenT<reco::MVAMEtPFCandInfoCollection> srcPFCandInfo_;
  edm::EDGetTokenT<reco::MVAMEtPFCandInfoCollection> srcPFCandInfoLeptonMatch_;
  typedef std::vector<edm::InputTag>  vInputTag;
  std::vector<edm::EDGetTokenT<edm::View<reco::Candidate> > > srcLeptons_;

  edm::EDGetTokenT<CorrMETData> srcType0Correction_;

  double sfNoPUjets_;
  double sfNoPUjetOffsetEnCorr_;
  double sfPUjets_;
  double sfNoPUunclChargedCands_;
  double sfPUunclChargedCands_;
  double sfUnclNeutralCands_;
  double sfType0Correction_;
  double sfLeptonIsoCones_;
  
  PFMEtSignInterfaceBase* pfMEtSignInterface_;
  double sfMEtCovMin_;
  double sfMEtCovMax_;

  bool saveInputs_;

  int verbosity_;
};

#endif
