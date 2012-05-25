#ifndef RecoMET_METProducers_PFMETProducerMVAData_h
#define RecoMET_METProducers_PFMETProducerMVAData_h

/** \class PFMETProducerMVAData
 *
 * Produce input objects used to compute MVA PFMET
 *
 * \authors Phil Harris, CERN
 *          Christian Veelken, LLR
 *
 * \version $Revision: 1.1 $
 *
 * $Id: PFMETProducerMVAData.h,v 1.1 2012/05/24 07:58:01 veelken Exp $
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

#include "DataFormats/METReco/interface/MVAMETData.h"
#include "DataFormats/METReco/interface/MVAMETDataFwd.h"
#include "RecoMET/METAlgorithms/interface/mvaMEtUtilities.h"

#include "RecoMET/METAlgorithms/interface/METAlgo.h"
#include "RecoMET/METAlgorithms/interface/PFSpecificAlgo.h"

#include "PhysicsTools/SelectorUtils/interface/PFJetIDSelectionFunctor.h"
//#include "CMGTools/External/interface/PileupJetIdAlgo.h"

#include <vector>

namespace reco
{
  class PFMETProducerMVAData : public edm::EDProducer
  {
   public:

    PFMETProducerMVAData(const edm::ParameterSet&);
    ~PFMETProducerMVAData();

   private:

    void produce(edm::Event&, const edm::EventSetup&);

    // auxiliary functions
    std::auto_ptr<reco::JetInfoCollection> computeJetInfo(
      const reco::PFJetCollection&, const reco::PFJetCollection&, const reco::VertexCollection&, const reco::Vertex*, double);
    std::auto_ptr<std::vector<mvaMEtUtilities::pfCandInfo> > computePFCandidateInfo(
      const reco::PFCandidateCollection&, const reco::Vertex*);
    std::auto_ptr<std::vector<reco::Vertex::Point> > computeVertexInfo(
      const reco::VertexCollection&);

    // configuration parameter
    edm::InputTag srcCorrJets_;
    edm::InputTag srcUncorrJets_;
    edm::InputTag srcPFCandidates_;
    edm::InputTag srcVertices_;
    edm::InputTag srcRho_;

    double globalThreshold_;

    double minCorrJetPt_;

    METAlgo metAlgo_;
    PFSpecificAlgo pfMEtSpecificAlgo_;

    PFJetIDSelectionFunctor* looseJetIdAlgo_;
    //PileupJetIdAlgo mvaJetIdAlgo_;

    int verbosity_;
  };
}

#endif
