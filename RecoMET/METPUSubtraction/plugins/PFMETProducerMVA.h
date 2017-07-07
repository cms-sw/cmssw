#ifndef RecoMET_METPUSubtraction_PFMETProducerMVA_h
#define RecoMET_METPUSubtraction_PFMETProducerMVA_h

/** \class PFMETProducerMVA
 *
 * Produce PFMET objects computed by MVA 
 *
 * \authors Phil Harris, CERN
 *          Christian Veelken, LLR
 *
 */

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"


#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/Candidate/interface/CandidateFwd.h"
#include "DataFormats/JetReco/interface/PFJet.h"
#include "DataFormats/JetReco/interface/PFJetCollection.h"
#include "DataFormats/Math/interface/deltaR.h"
#include "DataFormats/METReco/interface/CommonMETData.h"
#include "DataFormats/METReco/interface/PFMET.h"
#include "DataFormats/METReco/interface/PFMETCollection.h"
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/PatCandidates/interface/PackedCandidate.h"
#include "DataFormats/PatCandidates/interface/Tau.h"
#include "DataFormats/VertexReco/interface/Vertex.h"

#include "JetMETCorrections/JetCorrector/interface/JetCorrector.h"

#include "RecoMET/METAlgorithms/interface/METAlgo.h"
#include "RecoMET/METAlgorithms/interface/PFSpecificAlgo.h"
#include "RecoMET/METPUSubtraction/interface/PFMETAlgorithmMVA.h"
#include "RecoMET/METPUSubtraction/interface/MvaMEtUtilities.h"

#include "RecoJets/JetProducers/interface/PileupJetIdAlgo.h"
#include <TLorentzVector.h>

namespace reco
{
  class PFMETProducerMVA : public edm::stream::EDProducer<>
  {
   public:

    PFMETProducerMVA(const edm::ParameterSet&); 
    ~PFMETProducerMVA() override;

   private:
  
    void produce(edm::Event&, const edm::EventSetup&) override;

    // auxiliary functions
    std::vector<reco::PUSubMETCandInfo> computeLeptonInfo(const std::vector<edm::EDGetTokenT<reco::CandidateView > >& srcLeptons_,
							  const reco::CandidateView& pfCandidates,
							  const reco::Vertex* hardScatterVertex, 
							  int& lId, bool& lHasPhotons, edm::Event & iEvent);

    std::vector<reco::PUSubMETCandInfo> computeJetInfo(const reco::PFJetCollection&, const edm::Handle<reco::PFJetCollection>&,
							  const edm::ValueMap<float>&, const reco::VertexCollection&, 
							  const reco::Vertex*, const reco::JetCorrector &iCorr,
							  edm::Event & iEvent,const edm::EventSetup &iSetup,
							  std::vector<reco::PUSubMETCandInfo> &iLeptons,
							  std::vector<reco::PUSubMETCandInfo> &iCands);
    
    std::vector<reco::PUSubMETCandInfo> computePFCandidateInfo(const reco::CandidateView&, const reco::Vertex*);
    std::vector<reco::Vertex::Point> computeVertexInfo(const reco::VertexCollection&);
    double chargedEnFrac(const reco::Candidate *iCand,const reco::CandidateView& pfCandidates,const reco::Vertex* hardScatterVertex);
    
    bool   passPFLooseId(const reco::PFJet *iJet);
    bool   istau        (const reco::Candidate *iCand);
    double chargedFracInCone(const reco::Candidate *iCand,const reco::CandidateView& pfCandidates,const reco::Vertex* hardScatterVertex,double iDRMax=0.2);

   // configuration parameter
    edm::EDGetTokenT<reco::PFJetCollection> srcCorrJets_;
    edm::EDGetTokenT<reco::PFJetCollection> srcUncorrJets_;
    edm::EDGetTokenT<edm::ValueMap<float> > srcJetIds_;
    //edm::EDGetTokenT<reco::PFCandidateCollection> srcPFCandidates_;
    edm::EDGetTokenT<edm::View<reco::Candidate> > srcPFCandidatesView_;
    edm::EDGetTokenT<reco::VertexCollection> srcVertices_;
    edm::EDGetTokenT<reco::JetCorrector> mJetCorrector_;
    typedef std::vector<edm::InputTag> vInputTag;
    std::vector<edm::EDGetTokenT<reco::CandidateView > > srcLeptons_;
    int minNumLeptons_; // CV: option to skip MVA MET computation in case there are less than specified number of leptons in the event

    bool useType1_;
    
    double globalThreshold_;

    double minCorrJetPt_;

    METAlgo metAlgo_;
    PFSpecificAlgo pfMEtSpecificAlgo_;
    PFMETAlgorithmMVA mvaMEtAlgo_;
    bool mvaMEtAlgo_isInitialized_;
    // PileupJetIdAlgo mvaJetIdAlgo_;

    int verbosity_;
  };
}

#endif
