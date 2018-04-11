/* class PFTauMiniAODPrimaryVertexProducer
 * EDProducer of the 
 * authors: Ian M. Nugent
 * This work is based on the impact parameter work by Rosamaria Venditti and reconstructing the 3 prong taus.
 * The idea of the fully reconstructing the tau using a kinematic fit comes from
 * Lars Perchalla and Philip Sauerland Theses under Achim Stahl supervision. This
 * work was continued by Ian M. Nugent and Vladimir Cherepanov.
 * Thanks goes to Christian Veelken and Evan Klose Friis for their help and suggestions.
 */


#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Exception.h"

#include "TrackingTools/TransientTrack/interface/TransientTrackBuilder.h"
#include "TrackingTools/Records/interface/TransientTrackRecord.h"
#include "RecoVertex/VertexPrimitives/interface/TransientVertex.h"
#include "RecoVertex/AdaptiveVertexFit/interface/AdaptiveVertexFitter.h"
#include "RecoTauTag/RecoTau/interface/RecoTauVertexAssociator.h"

#include "DataFormats/TauReco/interface/PFTau.h"
#include "DataFormats/TauReco/interface/PFTauFwd.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrack.h" 
#include "DataFormats/GsfTrackReco/interface/GsfTrackFwd.h" 
#include "DataFormats/Common/interface/RefToBase.h"
#include "DataFormats/EgammaCandidates/interface/Electron.h"
#include "DataFormats/EgammaCandidates/interface/ElectronFwd.h"
#include "DataFormats/PatCandidates/interface/PackedCandidate.h"

#include "DataFormats/Common/interface/Association.h"
#include "DataFormats/Common/interface/AssociationVector.h"
#include "DataFormats/Common/interface/RefProd.h"

#include "DataFormats/TauReco/interface/PFTauDiscriminator.h"
#include "CommonTools/Utils/interface/StringCutObjectSelector.h"
#include <memory>
#include <boost/foreach.hpp>
#include <TFormula.h>
#include "DataFormats/Math/interface/deltaR.h"

#include <memory>

using namespace reco;
using namespace edm;
using namespace std;

class PFTauMiniAODPrimaryVertexProducer final : public edm::stream::EDProducer<> {
 public:
  enum Alg{useInputPV=0, useFrontPV};

  struct DiscCutPair{
    DiscCutPair():discr_(nullptr),cutFormula_(nullptr){}
    ~DiscCutPair(){delete cutFormula_;}
    const reco::PFTauDiscriminator* discr_;
    edm::EDGetTokenT<reco::PFTauDiscriminator> inputToken_;
    double cut_;
    TFormula* cutFormula_;
  };
  typedef std::vector<DiscCutPair*> DiscCutPairVec;

  explicit PFTauMiniAODPrimaryVertexProducer(const edm::ParameterSet& iConfig);
  ~PFTauMiniAODPrimaryVertexProducer() override;
  void produce(edm::Event&,const edm::EventSetup&) override;

 private:
  void nonTauTracksInPVFromPackedCands(const pat::PackedCandidateCollection&,
				       const size_t&,
				       const std::vector<const reco::Track*>&,
				       std::vector<const reco::Track*> &);

  edm::EDGetTokenT<std::vector<reco::PFTau> > pftauToken_;
  edm::EDGetTokenT<edm::View<reco::Electron> > electronToken_;
  edm::EDGetTokenT<edm::View<reco::Muon> > muonToken_;
  edm::EDGetTokenT<reco::VertexCollection> pvToken_;
  edm::EDGetTokenT<reco::BeamSpot> beamSpotToken_;
  edm::EDGetTokenT<pat::PackedCandidateCollection> packedCandsToken_, lostCandsToken_;
  int algorithm_;
  edm::ParameterSet qualityCutsPSet_;
  bool useBeamSpot_;
  bool useSelectedTaus_;
  bool removeMuonTracks_;
  bool removeElectronTracks_;
  DiscCutPairVec discriminators_;
  std::auto_ptr<StringCutObjectSelector<reco::PFTau> > cut_;
  std::auto_ptr<tau::RecoTauVertexAssociator> vertexAssociator_;
};

PFTauMiniAODPrimaryVertexProducer::PFTauMiniAODPrimaryVertexProducer(const edm::ParameterSet& iConfig):
  pftauToken_(consumes<std::vector<reco::PFTau> >(iConfig.getParameter<edm::InputTag>("PFTauTag"))),
  electronToken_(consumes<edm::View<reco::Electron> >(iConfig.getParameter<edm::InputTag>("ElectronTag"))),
  muonToken_(consumes<edm::View<reco::Muon> >(iConfig.getParameter<edm::InputTag>("MuonTag"))),
  pvToken_(consumes<reco::VertexCollection>(iConfig.getParameter<edm::InputTag>("PVTag"))),
  beamSpotToken_(consumes<reco::BeamSpot>(iConfig.getParameter<edm::InputTag>("beamSpot"))),
  packedCandsToken_(consumes<pat::PackedCandidateCollection>(iConfig.getParameter<edm::InputTag>("packedCandidatesTag"))),
  lostCandsToken_(consumes<pat::PackedCandidateCollection>(iConfig.getParameter<edm::InputTag>("lostCandidatesTag"))),
  algorithm_(iConfig.getParameter<int>("Algorithm")),
  qualityCutsPSet_(iConfig.getParameter<edm::ParameterSet>("qualityCuts")),
  useBeamSpot_(iConfig.getParameter<bool>("useBeamSpot")),
  useSelectedTaus_(iConfig.getParameter<bool>("useSelectedTaus")),
  removeMuonTracks_(iConfig.getParameter<bool>("RemoveMuonTracks")),
  removeElectronTracks_(iConfig.getParameter<bool>("RemoveElectronTracks"))
{
  /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  std::vector<edm::ParameterSet> discriminators =iConfig.getParameter<std::vector<edm::ParameterSet> >("discriminators");
  // Build each of our cuts
  BOOST_FOREACH(const edm::ParameterSet &pset, discriminators) {
    DiscCutPair* newCut = new DiscCutPair();
    newCut->inputToken_ =consumes<reco::PFTauDiscriminator>(pset.getParameter<edm::InputTag>("discriminator"));

    if ( pset.existsAs<std::string>("selectionCut") ) newCut->cutFormula_ = new TFormula("selectionCut", pset.getParameter<std::string>("selectionCut").data());
    else newCut->cut_ = pset.getParameter<double>("selectionCut");
    discriminators_.push_back(newCut);
  }
  // Build a string cut if desired
  if (iConfig.exists("cut")) cut_.reset(new StringCutObjectSelector<reco::PFTau>(iConfig.getParameter<std::string>( "cut" )));
  //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  produces<edm::AssociationVector<PFTauRefProd, std::vector<reco::VertexRef> > >();
  produces<VertexCollection>("PFTauPrimaryVertices"); 

  vertexAssociator_.reset(new tau::RecoTauVertexAssociator(qualityCutsPSet_,consumesCollector()));
}

PFTauMiniAODPrimaryVertexProducer::~PFTauMiniAODPrimaryVertexProducer(){}

namespace {
  const reco::Track* getTrack(const reco::Candidate& cand) {
    const pat::PackedCandidate* pCand = dynamic_cast<const pat::PackedCandidate*>(&cand);
    if (pCand && pCand->hasTrackDetails())
      return &pCand->pseudoTrack();
    return nullptr;
  }
}

void PFTauMiniAODPrimaryVertexProducer::produce(edm::Event& iEvent,const edm::EventSetup& iSetup){
  // Obtain Collections
  edm::ESHandle<TransientTrackBuilder> transTrackBuilder;
  iSetup.get<TransientTrackRecord>().get("TransientTrackBuilder",transTrackBuilder);
  
  edm::Handle<std::vector<reco::PFTau> > pfTaus;
  iEvent.getByToken(pftauToken_,pfTaus);

  edm::Handle<edm::View<reco::Electron> > electrons;
  iEvent.getByToken(electronToken_,electrons);

  edm::Handle<edm::View<reco::Muon> > muons;
  iEvent.getByToken(muonToken_,muons);

  edm::Handle<reco::VertexCollection > vertices;
  iEvent.getByToken(pvToken_,vertices);

  edm::Handle<reco::BeamSpot> beamSpot;
  iEvent.getByToken(beamSpotToken_,beamSpot);

  edm::Handle<pat::PackedCandidateCollection>  packedCands;
  iEvent.getByToken(packedCandsToken_, packedCands);

  edm::Handle<pat::PackedCandidateCollection>  lostCands;
  iEvent.getByToken(lostCandsToken_, lostCands);

  // Set Association Map
  auto avPFTauPV = std::make_unique<edm::AssociationVector<PFTauRefProd, std::vector<reco::VertexRef>>>(PFTauRefProd(pfTaus));
  auto vertexCollection_out = std::make_unique<VertexCollection>();
  reco::VertexRefProd vertexRefProd_out = iEvent.getRefBeforePut<reco::VertexCollection>("PFTauPrimaryVertices");

  // Load each discriminator
  BOOST_FOREACH(DiscCutPair *disc, discriminators_) {
    edm::Handle<reco::PFTauDiscriminator> discr;
    iEvent.getByToken(disc->inputToken_, discr);
    disc->discr_ = &(*discr);
  }

  // Set event for VerexAssociator if needed
  if(useInputPV==algorithm_)
    vertexAssociator_->setEvent(iEvent);

  // For each Tau Run Algorithim 
  if(pfTaus.isValid()){
    for(reco::PFTauCollection::size_type iPFTau = 0; iPFTau < pfTaus->size(); iPFTau++) {
      reco::PFTauRef tau(pfTaus, iPFTau);
      reco::Vertex thePV;
      size_t thePVkey = 0;
      if(useInputPV==algorithm_){
	reco::VertexRef thePVRef = vertexAssociator_->associatedVertex(*tau); 
	thePV = *thePVRef;
	thePVkey = thePVRef.key();
      }
      else if(useFrontPV==algorithm_){
	thePV=vertices->front();
	thePVkey = 0;
      }
      ///////////////////////
      // Check if it passed all the discrimiantors
      bool passed(true); 
      BOOST_FOREACH(const DiscCutPair* disc, discriminators_) {
        // Check this discriminator passes
	bool passedDisc = true;
	if ( disc->cutFormula_ )passedDisc = (disc->cutFormula_->Eval((*disc->discr_)[tau]) > 0.5);
	else passedDisc = ((*disc->discr_)[tau] > disc->cut_);
        if ( !passedDisc ){passed = false; break;}
      }
      if (passed && cut_.get()){passed = (*cut_)(*tau);}
      if (passed){
	std::vector<const reco::Track*> signalTracks;
	for(reco::PFTauCollection::size_type jPFTau = 0; jPFTau < pfTaus->size(); jPFTau++) {
	  if(useSelectedTaus_ || iPFTau==jPFTau){
	    reco::PFTauRef pfTauRef(pfTaus, jPFTau);
	    ///////////////////////////////////////////////////////////////////////////////////////////////
	    // Get tracks from PFTau daughters
	    for(const auto& pfcand : pfTauRef->signalChargedHadrCands()) {
	      if(pfcand.isNull()) continue;
	      const reco::Track* track = getTrack(*pfcand);
	      if(track == nullptr) continue;
	      signalTracks.push_back(track);
	    }
	  }
	}
	// Get Muon tracks
	if(removeMuonTracks_){

	  if(muons.isValid()) {
	    for(const auto& muon: *muons){
	      if(muon.track().isNonnull()) signalTracks.push_back(muon.track().get());
	    }
	  }
	}
	// Get Electron Tracks
	if(removeElectronTracks_){
	  if(electrons.isValid()) {
	    for(const auto& electron: *electrons){
	      if(electron.gsfTrack().isNonnull()) signalTracks.push_back(electron.gsfTrack().get());
	    }
	  }
	}
	///////////////////////////////////////////////////////////////////////////////////////////////
	// Get Non-Tau tracks
	std::vector<const reco::Track*> nonTauTracks;
	//PackedCandidates first...
 	if(packedCands.isValid()) {
	  nonTauTracksInPVFromPackedCands(*packedCands,thePVkey,signalTracks,nonTauTracks);
	}
	//then lostCandidates
 	if(lostCands.isValid()) {
	  nonTauTracksInPVFromPackedCands(*lostCands,thePVkey,signalTracks,nonTauTracks);
	}
	///////////////////////////////////////////////////////////////////////////////////////////////
	// Refit the vertex
	TransientVertex transVtx;
	std::vector<reco::TransientTrack> transTracks;
	for(const auto track: nonTauTracks){
	  transTracks.push_back(transTrackBuilder->build(*track));
	}
	bool fitOK(true);
	if ( transTracks.size() >= 2 ) {
	  AdaptiveVertexFitter avf;
	  avf.setWeightThreshold(0.1); //weight per track. allow almost every fit, else --> exception
	  try {
	    if ( !useBeamSpot_ ){
	      transVtx = avf.vertex(transTracks);
	    } else {
	      transVtx = avf.vertex(transTracks, *beamSpot);
	    }
	  } catch (...) {
	    fitOK = false;
	  }
	} else fitOK = false;
	if ( fitOK ) thePV = transVtx;
      }
      VertexRef vtxRef = reco::VertexRef(vertexRefProd_out, vertexCollection_out->size());
      vertexCollection_out->push_back(thePV);
      avPFTauPV->setValue(iPFTau, vtxRef);
    }
  }
  iEvent.put(std::move(vertexCollection_out),"PFTauPrimaryVertices");
  iEvent.put(std::move(avPFTauPV));
}

void PFTauMiniAODPrimaryVertexProducer::nonTauTracksInPVFromPackedCands(const pat::PackedCandidateCollection &cands,
									const size_t &thePVkey,
									const std::vector<const reco::Track*> &tauTracks,
									std::vector<const reco::Track*> &nonTauTracks){

  //Find candidates/tracks associated to thePVe
  for(auto const& cand: cands){
    if(cand.vertexRef().isNull()) continue;
    int quality = cand.pvAssociationQuality();
    if(cand.vertexRef().key()!=thePVkey ||
       (quality!=pat::PackedCandidate::UsedInFitTight &&
	quality!=pat::PackedCandidate::UsedInFitLoose)) continue;
    const reco::Track* track = getTrack(cand);
    if(track == nullptr) continue;
    //Remove signal (tau) tracks
    //MB: Only deltaR deltaPt overlap removal possible (?)
    //MB: It should be fine as pat objects stores same track info with same presision 
    bool skipTrack = false;
    for(auto const& tauTrack: tauTracks){
      if(deltaR2(tauTrack->eta(),tauTrack->phi(),
		 track->eta(),track->phi())<0.005*0.005
	 && std::abs(tauTrack->pt()/track->pt()-1.)<0.005
	 ){
	skipTrack = true;
	break;
      }
    }
    if(skipTrack) continue;
    nonTauTracks.push_back(track);
  }
}
  
DEFINE_FWK_MODULE(PFTauMiniAODPrimaryVertexProducer);
