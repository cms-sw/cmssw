/* class PFBaseTauPrimaryVertexProducer
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

#include "DataFormats/TauReco/interface/PFBaseTau.h"
#include "DataFormats/TauReco/interface/PFBaseTauFwd.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include "DataFormats/PatCandidates/interface/Muon.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrack.h" 
#include "DataFormats/GsfTrackReco/interface/GsfTrackFwd.h" 
#include "DataFormats/Common/interface/RefToBase.h"
#include "DataFormats/PatCandidates/interface/Electron.h"
#include "DataFormats/PatCandidates/interface/PackedCandidate.h"

#include "DataFormats/Common/interface/Association.h"
#include "DataFormats/Common/interface/AssociationVector.h"
#include "DataFormats/Common/interface/RefProd.h"

#include "DataFormats/TauReco/interface/PFBaseTauDiscriminator.h"
#include "CommonTools/Utils/interface/StringCutObjectSelector.h"
#include <memory>
#include <boost/foreach.hpp>
#include <TFormula.h>

#include <memory>

using namespace reco;
using namespace edm;
using namespace std;

class PFBaseTauPrimaryVertexProducer final : public edm::stream::EDProducer<> {
 public:
  enum Alg{useInputPV=0, useFontPV};

  struct DiscCutPair{
    DiscCutPair():discr_(nullptr),cutFormula_(nullptr){}
    ~DiscCutPair(){delete cutFormula_;}
    const reco::PFBaseTauDiscriminator* discr_;
    edm::EDGetTokenT<reco::PFBaseTauDiscriminator> inputToken_;
    double cut_;
    TFormula* cutFormula_;
  };
  typedef std::vector<DiscCutPair*> DiscCutPairVec;

  explicit PFBaseTauPrimaryVertexProducer(const edm::ParameterSet& iConfig);
  ~PFBaseTauPrimaryVertexProducer() override;
  void produce(edm::Event&,const edm::EventSetup&) override;

 private:
  edm::EDGetTokenT<std::vector<reco::PFBaseTau> > PFTauToken_;
  edm::EDGetTokenT<std::vector<pat::Electron> > ElectronToken_;
  edm::EDGetTokenT<std::vector<pat::Muon> > MuonToken_;
  edm::EDGetTokenT<reco::VertexCollection> PVToken_;
  edm::EDGetTokenT<reco::BeamSpot> beamSpotToken_;
  edm::EDGetTokenT<edm::View<pat::PackedCandidate> > packedCandsToken_, lostCandsToken_;
  int Algorithm_;
  edm::ParameterSet qualityCutsPSet_;
  bool useBeamSpot_;
  bool useSelectedTaus_;
  bool removeMuonTracks_;
  bool removeElectronTracks_;
  DiscCutPairVec discriminators_;
  std::auto_ptr<StringCutObjectSelector<reco::PFBaseTau> > cut_;
  std::auto_ptr<tau::RecoTauVertexAssociator> vertexAssociator_;
};

PFBaseTauPrimaryVertexProducer::PFBaseTauPrimaryVertexProducer(const edm::ParameterSet& iConfig):
  PFTauToken_(consumes<std::vector<reco::PFBaseTau> >(iConfig.getParameter<edm::InputTag>("PFTauTag"))),
  ElectronToken_(consumes<std::vector<pat::Electron> >(iConfig.getParameter<edm::InputTag>("ElectronTag"))),
  MuonToken_(consumes<std::vector<pat::Muon> >(iConfig.getParameter<edm::InputTag>("MuonTag"))),
  PVToken_(consumes<reco::VertexCollection>(iConfig.getParameter<edm::InputTag>("PVTag"))),
  beamSpotToken_(consumes<reco::BeamSpot>(iConfig.getParameter<edm::InputTag>("beamSpot"))),
  packedCandsToken_(consumes<edm::View<pat::PackedCandidate> >(iConfig.getParameter<edm::InputTag>("packedCandidatesTag"))),
  lostCandsToken_(consumes<edm::View<pat::PackedCandidate> >(iConfig.getParameter<edm::InputTag>("lostCandidatesTag"))),
  Algorithm_(iConfig.getParameter<int>("Algorithm")),
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
    newCut->inputToken_ =consumes<reco::PFBaseTauDiscriminator>(pset.getParameter<edm::InputTag>("discriminator"));

    if ( pset.existsAs<std::string>("selectionCut") ) newCut->cutFormula_ = new TFormula("selectionCut", pset.getParameter<std::string>("selectionCut").data());
    else newCut->cut_ = pset.getParameter<double>("selectionCut");
    discriminators_.push_back(newCut);
  }
  // Build a string cut if desired
  if (iConfig.exists("cut")) cut_.reset(new StringCutObjectSelector<reco::PFBaseTau>(iConfig.getParameter<std::string>( "cut" )));
  //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  produces<edm::AssociationVector<PFBaseTauRefProd, std::vector<reco::VertexRef> > >();
  produces<VertexCollection>("PFTauPrimaryVertices"); 

  vertexAssociator_.reset(new tau::RecoTauVertexAssociator(qualityCutsPSet_,consumesCollector()));
}

PFBaseTauPrimaryVertexProducer::~PFBaseTauPrimaryVertexProducer(){}

namespace {
  const reco::Track* getTrack(const reco::Candidate& cand) {
    const pat::PackedCandidate* pCand = dynamic_cast<const pat::PackedCandidate*>(&cand);
    if (pCand != nullptr && pCand->hasTrackDetails())
      return &pCand->pseudoTrack();
    return nullptr;
  }
}

void PFBaseTauPrimaryVertexProducer::produce(edm::Event& iEvent,const edm::EventSetup& iSetup){
  // Obtain Collections
  edm::ESHandle<TransientTrackBuilder> transTrackBuilder;
  iSetup.get<TransientTrackRecord>().get("TransientTrackBuilder",transTrackBuilder);
  
  edm::Handle<std::vector<reco::PFBaseTau> > Tau;
  iEvent.getByToken(PFTauToken_,Tau);

  edm::Handle<std::vector<pat::Electron> > Electron;
  iEvent.getByToken(ElectronToken_,Electron);

  edm::Handle<std::vector<pat::Muon> > Mu;
  iEvent.getByToken(MuonToken_,Mu);

  edm::Handle<reco::VertexCollection > PV;
  iEvent.getByToken(PVToken_,PV);

  edm::Handle<reco::BeamSpot> beamSpot;
  iEvent.getByToken(beamSpotToken_,beamSpot);

  edm::Handle<edm::View<pat::PackedCandidate> >  packedCands;
  iEvent.getByToken(packedCandsToken_, packedCands);

  edm::Handle<edm::View<pat::PackedCandidate> >  lostCands;
  iEvent.getByToken(lostCandsToken_, lostCands);

  // Set Association Map
  auto AVPFTauPV = std::make_unique<edm::AssociationVector<PFBaseTauRefProd, std::vector<reco::VertexRef>>>(PFBaseTauRefProd(Tau));
  auto VertexCollection_out = std::make_unique<VertexCollection>();
  reco::VertexRefProd VertexRefProd_out = iEvent.getRefBeforePut<reco::VertexCollection>("PFTauPrimaryVertices");

  // Load each discriminator
  BOOST_FOREACH(DiscCutPair *disc, discriminators_) {
    edm::Handle<reco::PFBaseTauDiscriminator> discr;
    iEvent.getByToken(disc->inputToken_, discr);
    disc->discr_ = &(*discr);
  }

  // Set event for VerexAssociator if needed
  if(useInputPV==Algorithm_)
    vertexAssociator_->setEvent(iEvent);

  // For each Tau Run Algorithim 
  if(Tau.isValid()){
    for(reco::PFBaseTauCollection::size_type iPFTau = 0; iPFTau < Tau->size(); iPFTau++) {
      reco::PFBaseTauRef tau(Tau, iPFTau);
      reco::Vertex thePV;
      size_t thePVkey = 0;
      if(useInputPV==Algorithm_){
	reco::VertexRef thePVRef = vertexAssociator_->associatedVertex(*tau); 
	thePV = *thePVRef;
	thePVkey = thePVRef.key();
      }
      else if(useFontPV==Algorithm_){
	thePV=PV->front();
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
	std::vector<const reco::Track*> SignalTracks;
	for(reco::PFBaseTauCollection::size_type jPFTau = 0; jPFTau < Tau->size(); jPFTau++) {
	  if(useSelectedTaus_ || iPFTau==jPFTau){
	    reco::PFBaseTauRef RefPFTau(Tau, jPFTau);
	    ///////////////////////////////////////////////////////////////////////////////////////////////
	    // Get tracks from PFTau daugthers
	    const std::vector<reco::CandidatePtr> cands = RefPFTau->signalPFChargedHadrCands(); 
	    for(std::vector<reco::CandidatePtr>::const_iterator iter = cands.begin(); iter!=cands.end(); ++iter){
	      if((*iter).isNull()) continue;
	      const reco::Track* track = getTrack(**iter);
	      if(track == nullptr) continue;
	      SignalTracks.push_back(track);
	    }
	  }
	}
	// Get Muon tracks
	if(removeMuonTracks_){

	  if(Mu.isValid()) {
	    for(pat::MuonCollection::size_type iMuon = 0; iMuon< Mu->size(); iMuon++){
	      pat::MuonRef RefMuon(Mu, iMuon);
	      if(RefMuon->track().isNonnull()) SignalTracks.push_back(RefMuon->track().get());
	    }
	  }
	}
	// Get Electron Tracks
	if(removeElectronTracks_){
	  if(Electron.isValid()) {
	    for(pat::ElectronCollection::size_type iElectron = 0; iElectron<Electron->size(); iElectron++){
	      pat::ElectronRef RefElectron(Electron, iElectron);
	      if(RefElectron->gsfTrack().isNonnull()) SignalTracks.push_back(RefElectron->gsfTrack().get());
	    }
	  }
	}
	///////////////////////////////////////////////////////////////////////////////////////////////
	// Get Non-Tau tracks
	std::vector<const reco::Track*> nonTauTracks;
	//PackedCandidates first...
 	if(packedCands.isValid()) {
	  //Find candidates/tracks associated to thePV
	  for(size_t iCand=0; iCand<packedCands->size(); ++iCand){
	    if((*packedCands)[iCand].vertexRef().isNull()) continue;
	    int quality = (*packedCands)[iCand].pvAssociationQuality();
	    if((*packedCands)[iCand].vertexRef().key()!=thePVkey ||
	       (quality!=pat::PackedCandidate::UsedInFitTight &&
		quality!=pat::PackedCandidate::UsedInFitLoose)) continue;
	    const reco::Track* track = getTrack((*packedCands)[iCand]);
	    if(track == nullptr) continue;
	    //Remove signal (tau) tracks
	    //MB: Only deltaR deltaPt overlap removal possible (?)
	    //MB: It should be fine as pat objects stores same track info with same presision 
	    bool skipTrack = false;
	    for(size_t iSigTrk=0 ; iSigTrk<SignalTracks.size(); ++iSigTrk){
	      if(deltaR2(SignalTracks[iSigTrk]->eta(),SignalTracks[iSigTrk]->phi(),
			 track->eta(),track->phi())<0.005*0.005
		 && std::abs(SignalTracks[iSigTrk]->pt()/track->pt()-1.)<0.005
		 ){
		skipTrack = true;
		break;
	      }
	    }
	    if(skipTrack) continue;
	    nonTauTracks.push_back(track);
	  } 
	}
	//then lostCandidates
 	if(lostCands.isValid()) {
	  //Find candidates/tracks associated to thePV
	  for(size_t iCand=0; iCand<lostCands->size(); ++iCand){
	    if((*lostCands)[iCand].vertexRef().isNull()) continue;
	    int quality = (*lostCands)[iCand].pvAssociationQuality();
	    if((*lostCands)[iCand].vertexRef().key()!=thePVkey ||
	       (quality!=pat::PackedCandidate::UsedInFitTight &&
		quality!=pat::PackedCandidate::UsedInFitLoose)) continue;
	    const reco::Track* track = getTrack((*lostCands)[iCand]);
	    if(track == nullptr) continue;
	    //Remove signal (tau) tracks
	    //MB: Only deltaR deltaPt overlap removal possible (?)
	    //MB: It should be fine as pat objects stores same track info with same presision 
	    bool skipTrack = false;
	    for(size_t iSigTrk=0 ; iSigTrk<SignalTracks.size(); ++iSigTrk){
	      if(deltaR2(SignalTracks[iSigTrk]->eta(),SignalTracks[iSigTrk]->phi(),
			 track->eta(),track->phi())<0.005*0.005
		 && std::abs(SignalTracks[iSigTrk]->pt()/track->pt()-1.)<0.005
		 ){
		skipTrack = true;
		break;
	      }
	    }
	    if(skipTrack) continue;
	    nonTauTracks.push_back(track);
	  } 
	}
	///////////////////////////////////////////////////////////////////////////////////////////////
	// Refit the vertex
	TransientVertex transVtx;
	std::vector<reco::TransientTrack> transTracks;
	for(size_t iTrk=0 ; iTrk<nonTauTracks.size(); ++iTrk){
	  transTracks.push_back(transTrackBuilder->build(*(nonTauTracks[iTrk])));
	}
	bool FitOk(true);
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
	    FitOk = false;
	  }
	} else FitOk = false;
	if ( FitOk ) thePV = transVtx;
      }
      VertexRef VRef = reco::VertexRef(VertexRefProd_out, VertexCollection_out->size());
      VertexCollection_out->push_back(thePV);
      AVPFTauPV->setValue(iPFTau, VRef);
    }
  }
  iEvent.put(std::move(VertexCollection_out),"PFTauPrimaryVertices");
  iEvent.put(std::move(AVPFTauPV));
}
  
DEFINE_FWK_MODULE(PFBaseTauPrimaryVertexProducer);
