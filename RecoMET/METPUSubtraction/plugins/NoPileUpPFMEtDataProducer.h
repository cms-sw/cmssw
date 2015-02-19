#ifndef RecoMET_METPUSubtraction_NoPileUpPFMEtDataProducer_h
#define RecoMET_METPUSubtraction_NoPileUpPFMEtDataProducer_h

/** \class NoPileUpPFMEtDataProducer
 *
 * Produce input objects used to compute MVA/No-PU MET
 *
 * \authors Phil Harris, CERN
 *          Christian Veelken, LLR
 *
 */

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "PhysicsTools/SelectorUtils/interface/PFJetIDSelectionFunctor.h"
#include "DataFormats/JetReco/interface/PileupJetIdentifier.h" 

#include "RecoMET/METPUSubtraction/interface/PFMEtSignInterfaceBase.h"
#include "RecoMET/METPUSubtraction/interface/NoPileUpMEtAuxFunctions.h"

#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/Candidate/interface/CandidateFwd.h"
#include "DataFormats/JetReco/interface/PFJet.h"
#include "DataFormats/JetReco/interface/PFJetCollection.h"
#include "DataFormats/Common/interface/ValueMap.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidateFwd.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Math/interface/deltaR.h"
#include "DataFormats/METReco/interface/PUSubMETData.h"       
#include "DataFormats/METReco/interface/PUSubMETDataFwd.h"    
#include "DataFormats/METReco/interface/SigInputObj.h"    //PH: preserve 5_3_x dependence

#include "JetMETCorrections/Objects/interface/JetCorrector.h"

class NoPileUpPFMEtDataProducer : public edm::stream::EDProducer<>
{
 public:
  
  NoPileUpPFMEtDataProducer(const edm::ParameterSet&);
  ~NoPileUpPFMEtDataProducer();
  
 private:
  
  void produce(edm::Event&, const edm::EventSetup&);
  
  std::string moduleLabel_;

  edm::EDGetTokenT<reco::PFJetCollection> srcJets_;
  edm::EDGetTokenT<edm::ValueMap<int> > srcJetIds_;
  double minJetPt_;
  PileupJetIdentifier::Id jetIdSelection_;
  std::string jetEnOffsetCorrLabel_;
  
  edm::EDGetTokenT<reco::PFCandidateCollection> srcPFCandidates_;
  edm::EDGetTokenT<edm::View<reco::PFCandidate> > srcPFCandidatesView_;
  edm::EDGetTokenT<PFCandToVertexAssMap> srcPFCandToVertexAssociations_;
  edm::EDGetTokenT<reco::PFJetCollection> srcJetsForMEtCov_;
  double minJetPtForMEtCov_;
  edm::EDGetTokenT<reco::VertexCollection> srcHardScatterVertex_;
  double dZcut_;
  
  PFJetIDSelectionFunctor* looseJetIdAlgo_;
  
  PFMEtSignInterfaceBase* pfMEtSignInterface_;

  int maxWarnings_;
  int numWarnings_;

  int verbosity_;
};

#endif
