#include "RecoMET/METProducers/interface/PFMETProducerMVA.h"

#include "FWCore/Utilities/interface/Exception.h"

#include "RecoMET/METAlgorithms/interface/METAlgo.h" 
#include "RecoMET/METAlgorithms/interface/PFSpecificAlgo.h"

#include "DataFormats/METReco/interface/PFMET.h"
#include "DataFormats/METReco/interface/PFMETCollection.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrack.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrackFwd.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"

#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/Common/interface/View.h"
#include "DataFormats/METReco/interface/CommonMETData.h"

#include <TMatrixD.h>

#include <algorithm>

using namespace reco;

PFMETProducerMVA::PFMETProducerMVA(const edm::ParameterSet& cfg) 
  : mvaMEtAlgo_(cfg),
    looseJetIdAlgo_(0)
    //mvaJetIdAlgo_(cfg)
{
  srcCorrJets_     = cfg.getParameter<edm::InputTag>("srcCorrJets");
  srcUncorrJets_   = cfg.getParameter<edm::InputTag>("srcUncorrJets");
  srcPFCandidates_ = cfg.getParameter<edm::InputTag>("srcPFCandidates");
  srcVertices_     = cfg.getParameter<edm::InputTag>("srcVertices");
  srcLeptons_      = cfg.getParameter<vInputTag>("srcLeptons");
  srcRho_          = cfg.getParameter<edm::InputTag>("srcRho");

  globalThreshold_ = cfg.getParameter<double>("globalThreshold");

  minCorrJetPt_    = cfg.getParameter<double>("minCorrJetPt");

  edm::ParameterSet cfgPFJetIdAlgo;
  cfgPFJetIdAlgo.addParameter<std::string>("version", "FIRSTDATA");
  cfgPFJetIdAlgo.addParameter<std::string>("quality", "LOOSE");
  looseJetIdAlgo_ = new PFJetIDSelectionFunctor(cfgPFJetIdAlgo);

  produces<reco::PFMETCollection>();
}

PFMETProducerMVA::~PFMETProducerMVA()
{
  delete looseJetIdAlgo_;
}

void PFMETProducerMVA::produce(edm::Event& evt, const edm::EventSetup& es) 
{  
  // get jets (corrected and uncorrected)
  edm::Handle<reco::PFJetCollection> corrJets;
  evt.getByLabel(srcCorrJets_, corrJets);

  edm::Handle<reco::PFJetCollection> uncorrJets;
  evt.getByLabel(srcUncorrJets_, uncorrJets);

  // get PFCandidates
  edm::Handle<reco::PFCandidateCollection> pfCandidates;
  evt.getByLabel(srcPFCandidates_, pfCandidates);

  typedef edm::View<reco::Candidate> CandidateView;
  edm::Handle<CandidateView> pfCandidates_view;
  evt.getByLabel(srcPFCandidates_, pfCandidates_view);

  // get leptons
  // (excluded from sum over PFCandidates when computing hadronic recoil)
  std::vector<reco::Candidate::LorentzVector> leptonInfo;
  for ( vInputTag::const_iterator srcLeptons_i = srcLeptons_.begin();
	srcLeptons_i != srcLeptons_.end(); ++srcLeptons_i ) {
    edm::Handle<CandidateView> leptons;
    evt.getByLabel(*srcLeptons_i, leptons);
    for ( CandidateView::const_iterator lepton = leptons->begin();
	  lepton != leptons->end(); ++lepton ) {
      leptonInfo.push_back(lepton->p4());
    }
  }

  // get vertices
  edm::Handle<reco::VertexCollection> vertices;
  evt.getByLabel(srcVertices_, vertices); 
  // take vertex with highest sum(trackPt) as the vertex of the "hard scatter" interaction
  // (= first entry in vertex collection)
  const reco::Vertex* hardScatterVertex = ( vertices->size() >= 1 ) ?
    &(vertices->front()) : 0;

  // get average energy density in the event
  edm::Handle<double> rho;
  evt.getByLabel(srcRho_, rho);

  // reconstruct "standard" particle-flow missing Et
  CommonMETData pfMEt_data;
  metAlgo_.run(pfCandidates_view, &pfMEt_data, globalThreshold_);
  reco::PFMET pfMEt = pfMEtSpecificAlgo_.addInfo(pfCandidates_view, pfMEt_data);

  // compute objects specific to MVA based MET reconstruction
  std::vector<mvaMEtUtilities::JetInfo> jetInfo = computeJetInfo(*uncorrJets, *corrJets, *vertices, hardScatterVertex, *rho);
  std::vector<mvaMEtUtilities::pfCandInfo> pfCandidateInfo = computePFCandidateInfo(*pfCandidates, hardScatterVertex);
  std::vector<reco::Vertex::Point> vertexInfo = computeVertexInfo(*vertices);

  // compute MVA based MET and estimate of its uncertainty
  mvaMEtAlgo_.setInput(leptonInfo, jetInfo, pfCandidateInfo, vertexInfo);
  mvaMEtAlgo_.evaluateMVA();

  pfMEt.setP4(mvaMEtAlgo_.getMEt());
  pfMEt.setSignificanceMatrix(mvaMEtAlgo_.getMEtCov());

  // add PFMET object to the event
  std::auto_ptr<reco::PFMETCollection> pfMEtCollection(new reco::PFMETCollection());
  pfMEtCollection->push_back(pfMEt);

  evt.put(pfMEtCollection);
}

std::vector<mvaMEtUtilities::JetInfo> PFMETProducerMVA::computeJetInfo(const reco::PFJetCollection& uncorrJets, 
								       const reco::PFJetCollection& corrJets, 
								       const reco::VertexCollection& vertices,
								       const reco::Vertex* hardScatterVertex,
								       double rho)
{
  std::vector<mvaMEtUtilities::JetInfo> retVal;
  for ( reco::PFJetCollection::const_iterator uncorrJet = uncorrJets.begin();
	uncorrJet != uncorrJets.end(); ++uncorrJet ) {
    for ( reco::PFJetCollection::const_iterator corrJet = corrJets.begin();
	  corrJet != corrJets.end(); ++corrJet ) {
      if ( uncorrJet->jetArea() != corrJet->jetArea()        ) continue;
      if ( !(fabs(uncorrJet->eta() - corrJet->eta()) < 0.01) ) continue;
      if ( !(corrJet->pt() > minCorrJetPt_)                  ) continue; 
      pat::strbitset dummy;
      bool passesLooseJetId = (*looseJetIdAlgo_)(*corrJet);
      if ( !passesLooseJetId                                 ) continue; 
      double jetEnCorrFactor = corrJet->pt()/uncorrJet->pt();
      mvaMEtUtilities::JetInfo jetInfo;
      jetInfo.p4_            = corrJet->p4();
      //jetInfo.mva_           = mvaJetIdAlgo_.computeIdVariables(&(*corrJet), jetEnCorrFactor, hardScatterVertex, vertices, true).mva();
      // CV: maybe better divide by uncorrected energy ?
      //    (corrected jet energy can be negative sometimes)
      jetInfo.neutralEnFrac_ = (corrJet->neutralEmEnergy() + corrJet->neutralHadronEnergy())/corrJet->energy();

      // PH: apply rho correction
      //    (following with Jet MET convenors use rho correction only below 10 GeV)
      // CV: why apply rho correction, jet energies are already corrected by L1Fastjet + L2 + L3 (+ L2L3Residual), no ?
      if ( corrJet->pt() < 10. ) {
	jetInfo.p4_ *= std::max((corrJet->pt() - rho*corrJet->jetArea())/corrJet->pt(), 0.); // CV: needs to be uncorrJet->energy
	                                                                                     //     instead of corrJet->pt ?
      }
      retVal.push_back(jetInfo);
      break;
    }
  }
  return retVal;
}

std::vector<mvaMEtUtilities::pfCandInfo> PFMETProducerMVA::computePFCandidateInfo(const reco::PFCandidateCollection& pfCandidates,
										  const reco::Vertex* hardScatterVertex)
{
  std::vector<mvaMEtUtilities::pfCandInfo> retVal;
  for ( reco::PFCandidateCollection::const_iterator pfCandidate = pfCandidates.begin();
	pfCandidate != pfCandidates.end(); ++pfCandidate ) {
    double dZ = -999.; // PH: If no vertex is reconstructed in the event
                       //     or PFCandidate has no track, set dZ to -999
    if ( hardScatterVertex ) {
      if      ( pfCandidate->trackRef().isNonnull()    ) dZ = fabs(pfCandidate->trackRef()->dz(hardScatterVertex->position()));
      else if ( pfCandidate->gsfTrackRef().isNonnull() ) dZ = fabs(pfCandidate->gsfTrackRef()->dz(hardScatterVertex->position()));
    }
    mvaMEtUtilities::pfCandInfo pfCandidateInfo;
    pfCandidateInfo.p4_ = pfCandidate->p4();
    pfCandidateInfo.dZ_ = dZ;
    retVal.push_back(pfCandidateInfo);
  }
  return retVal;
}

std::vector<reco::Vertex::Point> PFMETProducerMVA::computeVertexInfo(const reco::VertexCollection& vertices)
{
  std::vector<reco::Vertex::Point> retVal;
  for ( reco::VertexCollection::const_iterator vertex = vertices.begin();
	vertex != vertices.end(); ++vertex ) {
    retVal.push_back(vertex->position());
  }
  return retVal;
}
