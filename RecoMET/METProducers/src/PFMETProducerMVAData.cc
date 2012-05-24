#include "RecoMET/METProducers/interface/PFMETProducerMVAData.h"

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
#include "DataFormats/METReco/interface/MVAMETData.h"
#include "DataFormats/METReco/interface/MVAMETDataFwd.h"

#include <TMatrixD.h>

#include <algorithm>

using namespace reco;

PFMETProducerMVAData::PFMETProducerMVAData(const edm::ParameterSet& cfg)
  : looseJetIdAlgo_(0),
    mvaJetIdAlgo_(cfg)
{
  srcCorrJets_     = cfg.getParameter<edm::InputTag>("srcCorrJets");
  srcUncorrJets_   = cfg.getParameter<edm::InputTag>("srcUncorrJets");
  srcPFCandidates_ = cfg.getParameter<edm::InputTag>("srcPFCandidates");
  srcVertices_     = cfg.getParameter<edm::InputTag>("srcVertices");
  srcRho_          = cfg.getParameter<edm::InputTag>("srcRho");

  globalThreshold_ = cfg.getParameter<double>("globalThreshold");

  minCorrJetPt_    = cfg.getParameter<double>("minCorrJetPt");

  edm::ParameterSet cfgPFJetIdAlgo;
  cfgPFJetIdAlgo.addParameter<std::string>("version", "FIRSTDATA");
  cfgPFJetIdAlgo.addParameter<std::string>("quality", "LOOSE");
  looseJetIdAlgo_ = new PFJetIDSelectionFunctor(cfgPFJetIdAlgo);

  verbosity_ = ( cfg.exists("verbosity") ) ?
    cfg.getParameter<int>("verbosity") : 0;

  if ( verbosity_ ) {
    std::cout << "<PFMETProducerMVAData::PFMETProducerMVAData>:" << std::endl;
    std::cout << " srcCorrJets = " << srcCorrJets_.label() << std::endl;
    std::cout << " srcUncorrJets = " << srcUncorrJets_.label() << std::endl;
    std::cout << " srcPFCandidates = " << srcPFCandidates_.label() << std::endl;
    std::cout << " srcVertices = " << srcVertices_.label() << std::endl;
    std::cout << " srcRho = " << srcVertices_.label() << std::endl;
  }

  produces<reco::PFMETCollection>();
  produces<reco::JetInfoCollection>();
  // Mapping of PFCands -> DZ from PV
  produces<edm::ValueMap<float> >();
  produces<std::vector<reco::Vertex::Point> >();
}

PFMETProducerMVAData::~PFMETProducerMVAData()
{
  delete looseJetIdAlgo_;
}

void PFMETProducerMVAData::produce(edm::Event& evt, const edm::EventSetup& es)
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

  // compute objects specific to MVA based MET reconstruction
  std::auto_ptr<reco::JetInfoCollection> jetInfo = computeJetInfo(*uncorrJets, *corrJets, *vertices, hardScatterVertex, *rho);
  std::auto_ptr<std::vector<mvaMEtUtilities::pfCandInfo> > pfCandidateInfo = computePFCandidateInfo(*pfCandidates, hardScatterVertex);

  // PFCandidate DZs to vertex
  std::vector<float> dzs;
  dzs.reserve(pfCandidateInfo->size());
  for (size_t i = 0; i < pfCandidateInfo->size(); ++i) {
    dzs.push_back(pfCandidateInfo->at(i).dZ_);
  }
  std::auto_ptr<edm::ValueMap<float> > dzMap(new edm::ValueMap<float>());
  edm::ValueMap<float>::Filler filler(*dzMap);
  filler.insert(pfCandidates, dzs.begin(), dzs.end());
  filler.fill();

  std::auto_ptr<std::vector<reco::Vertex::Point> > vertexInfo = computeVertexInfo(*vertices);

  // reconstruct "standard" particle-flow missing Et
  CommonMETData pfMEt_data;
  metAlgo_.run(pfCandidates_view, &pfMEt_data, globalThreshold_);
  reco::PFMET pfMEt = pfMEtSpecificAlgo_.addInfo(pfCandidates_view, pfMEt_data);

  std::auto_ptr<reco::PFMETCollection> pfMEtCollection(new reco::PFMETCollection());
  pfMEtCollection->push_back(pfMEt);

  // Put collections into the event
  evt.put(pfMEtCollection);
  evt.put(jetInfo);
  evt.put(dzMap);
  evt.put(vertexInfo);
}

std::auto_ptr<reco::JetInfoCollection> PFMETProducerMVAData::computeJetInfo(const reco::PFJetCollection& uncorrJets,
								       const reco::PFJetCollection& corrJets,
								       const reco::VertexCollection& vertices,
								       const reco::Vertex* hardScatterVertex,
								       double rho)
{
  std::auto_ptr<reco::JetInfoCollection> retVal(new reco::JetInfoCollection);
  for ( reco::PFJetCollection::const_iterator uncorrJet = uncorrJets.begin();
	uncorrJet != uncorrJets.end(); ++uncorrJet ) {
    for ( reco::PFJetCollection::const_iterator corrJet = corrJets.begin();
	  corrJet != corrJets.end(); ++corrJet ) {
      // match corrected and uncorrected jets
      if ( uncorrJet->jetArea() != corrJet->jetArea() ) continue;
      if ( !(fabs(uncorrJet->eta() - corrJet->eta()) < 0.01) ) continue;

      // check that jet passes loose PFJet id.
      bool passesLooseJetId = (*looseJetIdAlgo_)(*corrJet);
      if ( !passesLooseJetId ) continue;

      // compute jet energy correction factor
      // (= ratio of corrected/uncorrected jet Pt)
      double jetEnCorrFactor = corrJet->pt()/uncorrJet->pt();
      reco::JetInfo jetInfo;

      // PH: apply jet energy corrections for all Jets ignoring recommendations
      jetInfo.p4_ = corrJet->p4();

      // check that jet Pt used to compute MVA based jet id. is above threshold
      if ( !(jetInfo.p4_.pt() > minCorrJetPt_) ) continue;

      jetInfo.mva_ = mvaJetIdAlgo_.computeIdVariables(&(*corrJet), jetEnCorrFactor, hardScatterVertex, vertices, true).mva();
      jetInfo.neutralEnFrac_ = (uncorrJet->neutralEmEnergy() + uncorrJet->neutralHadronEnergy())/uncorrJet->energy();

      retVal->push_back(jetInfo);
      break;
    }
  }
  return retVal;
}

std::auto_ptr<std::vector<mvaMEtUtilities::pfCandInfo> > PFMETProducerMVAData::computePFCandidateInfo(const reco::PFCandidateCollection& pfCandidates,
										  const reco::Vertex* hardScatterVertex)
{
  std::auto_ptr<std::vector<mvaMEtUtilities::pfCandInfo> > retVal(new std::vector<mvaMEtUtilities::pfCandInfo> );
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
    retVal->push_back(pfCandidateInfo);
  }
  return retVal;
}

std::auto_ptr<std::vector<reco::Vertex::Point> > PFMETProducerMVAData::computeVertexInfo(const reco::VertexCollection& vertices)
{
  std::auto_ptr<std::vector<reco::Vertex::Point> > retVal(new std::vector<reco::Vertex::Point>());
  for ( reco::VertexCollection::const_iterator vertex = vertices.begin();
	vertex != vertices.end(); ++vertex ) {
    retVal->push_back(vertex->position());
  }
  return retVal;
}
