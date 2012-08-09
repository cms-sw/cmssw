#ifndef RecoMET_METProducers_PFMETProducerMVA_h
#define RecoMET_METProducers_PFMETProducerMVA_h

/** \class PFMETProducerMVA
 *
 * Produce PFMET objects computed by MVA 
 *
 * \authors Phil Harris, CERN
 *          Christian Veelken, LLR
 *
 * \version $Revision: 1.3 $
 *
 * $Id: PFMETProducerMVA.h,v 1.3 2012/05/02 10:29:52 veelken Exp $
 *
 */

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidateFwd.h"
#include "DataFormats/JetReco/interface/PFJet.h"
#include "DataFormats/JetReco/interface/PFJetCollection.h"

#include "RecoMET/METAlgorithms/interface/METAlgo.h"
#include "RecoMET/METAlgorithms/interface/PFSpecificAlgo.h"
#include "RecoMET/METAlgorithms/interface/PFMETAlgorithmMVA.h"
#include "RecoMET/METAlgorithms/interface/mvaMEtUtilities.h"

//#include "CMGTools/External/interface/PileupJetIdAlgo.h"

#include <vector>

namespace reco
{
  class PFMETProducerMVA : public edm::EDProducer
  {
   public:

    PFMETProducerMVA(const edm::ParameterSet&); 
    ~PFMETProducerMVA();

   private:
  
    void produce(edm::Event&, const edm::EventSetup&);

    // auxiliary functions
    std::vector<mvaMEtUtilities::JetInfo> computeJetInfo(
      const reco::PFJetCollection&, const reco::PFJetCollection&, const reco::VertexCollection&, const reco::Vertex*, double);
    std::vector<mvaMEtUtilities::pfCandInfo> computePFCandidateInfo(
      const reco::PFCandidateCollection&, const reco::Vertex*);
    std::vector<reco::Vertex::Point> computeVertexInfo(
      const reco::VertexCollection&);
    double chargedFrac(const reco::Candidate *iCand);
    bool   passPFLooseId(const reco::PFJet *iJet);
    bool   istau        (const reco::Candidate *iCand);
   // configuration parameter
    edm::InputTag srcCorrJets_;
    edm::InputTag srcUncorrJets_;
    edm::InputTag srcPFCandidates_;
    edm::InputTag srcVertices_;
    typedef std::vector<edm::InputTag> vInputTag;
    vInputTag srcLeptons_;
    edm::InputTag srcRho_;

    double globalThreshold_;

    double minCorrJetPt_;

    METAlgo metAlgo_;
    PFSpecificAlgo pfMEtSpecificAlgo_;
    PFMETAlgorithmMVA mvaMEtAlgo_;

    //PFJetIDSelectionFunctor* looseJetIdAlgo_;
    //PileupJetIdAlgo mvaJetIdAlgo_;

    int verbosity_;
  };
}

#endif
