/* class PFTauPrimaryVertexProducer
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

#include "DataFormats/Common/interface/Association.h"
#include "DataFormats/Common/interface/AssociationVector.h"
#include "DataFormats/Common/interface/RefProd.h"

#include "DataFormats/TauReco/interface/PFTauDiscriminator.h"
#include "CommonTools/Utils/interface/StringCutObjectSelector.h"
#include <memory>
#include <boost/foreach.hpp>
#include <TFormula.h>

#include <memory>

using namespace reco;
using namespace edm;
using namespace std;

class PFTauPrimaryVertexProducer final : public edm::stream::EDProducer<> {
 public:
  enum Alg{useInputPV=0, useFontPV};

  struct DiscCutPair{
    DiscCutPair():discr_(nullptr),cutFormula_(nullptr){}
    ~DiscCutPair(){delete cutFormula_;}
    const reco::PFTauDiscriminator* discr_;
    edm::EDGetTokenT<reco::PFTauDiscriminator> inputToken_;
    double cut_;
    TFormula* cutFormula_;
  };
  typedef std::vector<DiscCutPair*> DiscCutPairVec;

  explicit PFTauPrimaryVertexProducer(const edm::ParameterSet& iConfig);
  ~PFTauPrimaryVertexProducer() override;
  void produce(edm::Event&,const edm::EventSetup&) override;

 private:
  edm::InputTag PFTauTag_;
  edm::EDGetTokenT<std::vector<reco::PFTau> >PFTauToken_;
  edm::InputTag ElectronTag_;
  edm::EDGetTokenT<std::vector<reco::Electron> >ElectronToken_;
  edm::InputTag MuonTag_;
  edm::EDGetTokenT<std::vector<reco::Muon> >MuonToken_;
  edm::InputTag PVTag_;
  edm::EDGetTokenT<reco::VertexCollection> PVToken_;
  edm::InputTag beamSpotTag_;
  edm::EDGetTokenT<reco::BeamSpot> beamSpotToken_;
  int Algorithm_;
  edm::ParameterSet qualityCutsPSet_;
  bool useBeamSpot_;
  bool useSelectedTaus_;
  bool RemoveMuonTracks_;
  bool RemoveElectronTracks_;
  DiscCutPairVec discriminators_;
  std::unique_ptr<StringCutObjectSelector<reco::PFTau> > cut_;
  std::unique_ptr<tau::RecoTauVertexAssociator> vertexAssociator_;
};

PFTauPrimaryVertexProducer::PFTauPrimaryVertexProducer(const edm::ParameterSet& iConfig):
  PFTauTag_(iConfig.getParameter<edm::InputTag>("PFTauTag")),
  PFTauToken_(consumes<std::vector<reco::PFTau> >(PFTauTag_)),
  ElectronTag_(iConfig.getParameter<edm::InputTag>("ElectronTag")),
  ElectronToken_(consumes<std::vector<reco::Electron> >(ElectronTag_)),
  MuonTag_(iConfig.getParameter<edm::InputTag>("MuonTag")),
  MuonToken_(consumes<std::vector<reco::Muon> >(MuonTag_)),
  PVTag_(iConfig.getParameter<edm::InputTag>("PVTag")),
  PVToken_(consumes<reco::VertexCollection>(PVTag_)),
  beamSpotTag_(iConfig.getParameter<edm::InputTag>("beamSpot")),
  beamSpotToken_(consumes<reco::BeamSpot>(beamSpotTag_)),
  Algorithm_(iConfig.getParameter<int>("Algorithm")),
  qualityCutsPSet_(iConfig.getParameter<edm::ParameterSet>("qualityCuts")),
  useBeamSpot_(iConfig.getParameter<bool>("useBeamSpot")),
  useSelectedTaus_(iConfig.getParameter<bool>("useSelectedTaus")),
  RemoveMuonTracks_(iConfig.getParameter<bool>("RemoveMuonTracks")),
  RemoveElectronTracks_(iConfig.getParameter<bool>("RemoveElectronTracks"))
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

PFTauPrimaryVertexProducer::~PFTauPrimaryVertexProducer(){}

void PFTauPrimaryVertexProducer::produce(edm::Event& iEvent,const edm::EventSetup& iSetup){
  // Obtain Collections
  edm::ESHandle<TransientTrackBuilder> transTrackBuilder;
  iSetup.get<TransientTrackRecord>().get("TransientTrackBuilder",transTrackBuilder);
  
  edm::Handle<std::vector<reco::PFTau> > Tau;
  iEvent.getByToken(PFTauToken_,Tau);

  edm::Handle<std::vector<reco::Electron> > Electron;
  iEvent.getByToken(ElectronToken_,Electron);

  edm::Handle<std::vector<reco::Muon> > Mu;
  iEvent.getByToken(MuonToken_,Mu);

  edm::Handle<reco::VertexCollection > PV;
  iEvent.getByToken(PVToken_,PV);

  edm::Handle<reco::BeamSpot> beamSpot;
  iEvent.getByToken(beamSpotToken_,beamSpot);

  // Set Association Map
  auto AVPFTauPV = std::make_unique<edm::AssociationVector<PFTauRefProd, std::vector<reco::VertexRef>>>(PFTauRefProd(Tau));
  auto VertexCollection_out = std::make_unique<VertexCollection>();
  reco::VertexRefProd VertexRefProd_out = iEvent.getRefBeforePut<reco::VertexCollection>("PFTauPrimaryVertices");

  // Load each discriminator
  BOOST_FOREACH(DiscCutPair *disc, discriminators_) {
    edm::Handle<reco::PFTauDiscriminator> discr;
    iEvent.getByToken(disc->inputToken_, discr);
    disc->discr_ = &(*discr);
  }

  // Set event for VerexAssociator if needed
  if(useInputPV==Algorithm_)
    vertexAssociator_->setEvent(iEvent);

  // For each Tau Run Algorithim 
  if(Tau.isValid()){
    for(reco::PFTauCollection::size_type iPFTau = 0; iPFTau < Tau->size(); iPFTau++) {
      reco::PFTauRef tau(Tau, iPFTau);
      reco::Vertex thePV;
      if(useInputPV==Algorithm_){
	thePV =(*( vertexAssociator_->associatedVertex(*tau)));
      }
      else if(useFontPV==Algorithm_){
	thePV=PV->front();
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
	std::vector<reco::TrackBaseRef> SignalTracks;
	for(reco::PFTauCollection::size_type jPFTau = 0; jPFTau < Tau->size(); jPFTau++) {
	  if(useSelectedTaus_ || iPFTau==jPFTau){
	    reco::PFTauRef RefPFTau(Tau, jPFTau);
	    ///////////////////////////////////////////////////////////////////////////////////////////////
	    // Get tracks from PFTau daugthers
	    const std::vector<edm::Ptr<reco::PFCandidate> > cands = RefPFTau->signalPFChargedHadrCands(); 
	    for (std::vector<edm::Ptr<reco::PFCandidate> >::const_iterator iter = cands.begin(); iter!=cands.end(); iter++){
	      if(iter->get()->trackRef().isNonnull()) SignalTracks.push_back(reco::TrackBaseRef(iter->get()->trackRef()));
	      else if(iter->get()->gsfTrackRef().isNonnull()){SignalTracks.push_back(reco::TrackBaseRef(((iter)->get()->gsfTrackRef())));}
	    }
	  }
	}
	// Get Muon tracks
	if(RemoveMuonTracks_){

	  if(Mu.isValid()) {
	    for(reco::MuonCollection::size_type iMuon = 0; iMuon< Mu->size(); iMuon++){
	      reco::MuonRef RefMuon(Mu, iMuon);
	      if(RefMuon->track().isNonnull()) SignalTracks.push_back(reco::TrackBaseRef(RefMuon->track()));
	    }
	  }
	}
	// Get Electron Tracks
	if(RemoveElectronTracks_){
	  if(Electron.isValid()) {
	    for(reco::ElectronCollection::size_type iElectron = 0; iElectron<Electron->size(); iElectron++){
	      reco::ElectronRef RefElectron(Electron, iElectron);
	      if(RefElectron->track().isNonnull()) SignalTracks.push_back(reco::TrackBaseRef(RefElectron->track()));
	    }
	  }
	}
	///////////////////////////////////////////////////////////////////////////////////////////////
	// Get Non-Tau tracks
	reco::TrackCollection nonTauTracks;
	for(std::vector<reco::TrackBaseRef>::const_iterator vtxTrkRef=thePV.tracks_begin();vtxTrkRef<thePV.tracks_end();vtxTrkRef++){
	  bool matched = false;
	  for (unsigned int sigTrk = 0; sigTrk < SignalTracks.size(); sigTrk++) {
	    if ( (*vtxTrkRef) == SignalTracks[sigTrk] ) {
	      matched = true;
	    }
	  }
	  if ( !matched ) nonTauTracks.push_back(**vtxTrkRef);
	}   
	///////////////////////////////////////////////////////////////////////////////////////////////
	// Refit the vertex
	TransientVertex transVtx;
	std::vector<reco::TransientTrack> transTracks;
	for (reco::TrackCollection::iterator iter=nonTauTracks.begin(); iter!=nonTauTracks.end(); ++iter){
	  transTracks.push_back(transTrackBuilder->build(*iter));
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
  
DEFINE_FWK_MODULE(PFTauPrimaryVertexProducer);
