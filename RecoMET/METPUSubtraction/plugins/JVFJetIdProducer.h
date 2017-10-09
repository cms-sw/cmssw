#ifndef RecoMET_METPUSubtraction_MVAPFMEtDataProducer_h
#define RecoMET_METPUSubtraction_MVAPFMEtDataProducer_h

/** \class JVFJetIdProducer
 *
 * Discriminate jets originating from the hard-scatter event from pile-up jets,
 * based on the fraction of tracks within the jet that are associated to the hard-scatter vertex.
 * Jets outside the tracking acceptance are considered to originate from the hard-scatter event per default.
 * Optionally, they can be classified as pile-up.
 *
 * \authors Christian Veelken, LLR
 *
 */

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "PhysicsTools/SelectorUtils/interface/PFJetIDSelectionFunctor.h"
#include "DataFormats/JetReco/interface/PileupJetIdentifier.h" 

#include "RecoMET/METPUSubtraction/interface/NoPileUpMEtAuxFunctions.h"

class JVFJetIdProducer : public edm::stream::EDProducer<>
{
 public:
  
  JVFJetIdProducer(const edm::ParameterSet&);
  ~JVFJetIdProducer();
  
 private:
  
  void produce(edm::Event&, const edm::EventSetup&);
  
  edm::EDGetTokenT<reco::PFJetCollection > srcJets_;

  edm::EDGetTokenT<reco::PFCandidateCollection> srcPFCandidates_;
  edm::EDGetTokenT<PFCandToVertexAssMap> srcPFCandToVertexAssociations_;
  edm::EDGetTokenT<reco::VertexCollection> srcHardScatterVertex_;
  double minTrackPt_;
  double dZcut_;

  double JVFcut_;

  int neutralJetOption_;
  
  int verbosity_;
};

#endif
