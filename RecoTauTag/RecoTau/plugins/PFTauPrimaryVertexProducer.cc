/* class PFTauPrimaryVertexProducer
 * EDProducer of the 
 * authors: Ian M. Nugent
 */


#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Exception.h"

#include "TrackingTools/TransientTrack/interface/TransientTrackBuilder.h"
#include "TrackingTools/Records/interface/TransientTrackRecord.h"
#include "RecoVertex/VertexPrimitives/interface/TransientVertex.h"
#include "RecoVertex/AdaptiveVertexFit/interface/AdaptiveVertexFitter.h"

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

#include <memory>

using namespace reco;
using namespace edm;
using namespace std;

class PFTauPrimaryVertexProducer : public EDProducer {
 public:
  enum Alg{useInputPV=0, useFontPV};

  explicit PFTauPrimaryVertexProducer(const edm::ParameterSet& iConfig);
  ~PFTauPrimaryVertexProducer();
  virtual void produce(edm::Event&,const edm::EventSetup&);
 private:
  edm::InputTag PFTauTag_;
  edm::InputTag ElectronTag_;
  edm::InputTag MuonTag_;
  edm::InputTag PVTag_;
  edm::InputTag beamSpotTag_;
  edm::InputTag TrackCollectionTag_;
  edm::InputTag TracksTag_;
  int Algorithm_;
  bool useBeamSpot_;
  bool useSelectedTaus_;
  bool RemoveMuonTracks_;
  bool RemoveElectronTracks_;
};

PFTauPrimaryVertexProducer::PFTauPrimaryVertexProducer(const edm::ParameterSet& iConfig):
  PFTauTag_(iConfig.getParameter<edm::InputTag>("PFTauTag")),
  ElectronTag_(iConfig.getParameter<edm::InputTag>("ElectronTag")),
  MuonTag_(iConfig.getParameter<edm::InputTag>("MuonTag")),
  PVTag_(iConfig.getParameter<edm::InputTag>("PVTag")),
  beamSpotTag_(iConfig.getParameter<edm::InputTag>("beamSpot")),
  TrackCollectionTag_(iConfig.getParameter<edm::InputTag>("TrackCollectionTag")),
  Algorithm_(iConfig.getParameter<int>("Algorithm")),
  useBeamSpot_(iConfig.getParameter<bool>("useBeamSpot")),
  useSelectedTaus_(iConfig.getParameter<bool>("useSelectedTaus")),
  RemoveMuonTracks_(iConfig.getParameter<bool>("RemoveMuonTracks")),
  RemoveElectronTracks_(iConfig.getParameter<bool>("RemoveElectronTracks"))
{
  produces<edm::AssociationVector<PFTauRefProd, std::vector<reco::VertexRef> > >();
  produces<VertexCollection>("PFTauPrimaryVertices"); 
}

PFTauPrimaryVertexProducer::~PFTauPrimaryVertexProducer(){

}

void PFTauPrimaryVertexProducer::produce(edm::Event& iEvent,const edm::EventSetup& iSetup){
  // Obtain Collections
  edm::ESHandle<TransientTrackBuilder> transTrackBuilder;
  iSetup.get<TransientTrackRecord>().get("TransientTrackBuilder",transTrackBuilder);
  
  edm::Handle<std::vector<reco::PFTau> > Tau;
  iEvent.getByLabel(PFTauTag_,Tau);

  edm::Handle<std::vector<reco::Electron> > Electron;
  iEvent.getByLabel(ElectronTag_,Electron);

  edm::Handle<std::vector<reco::Muon> > Mu;
  iEvent.getByLabel(MuonTag_,Mu);

  edm::Handle<reco::VertexCollection > PV;
  iEvent.getByLabel(PVTag_,PV);

  edm::Handle<reco::BeamSpot> beamSpot;
  iEvent.getByLabel(beamSpotTag_,beamSpot);

  edm::Handle<reco::TrackCollection> trackCollection;
  iEvent.getByLabel(TrackCollectionTag_,trackCollection);

  // Set Association Map
  auto_ptr<edm::AssociationVector<PFTauRefProd, std::vector<reco::VertexRef> > > AVPFTauPV(new edm::AssociationVector<PFTauRefProd, std::vector<reco::VertexRef> >(PFTauRefProd(Tau)));
  std::auto_ptr<VertexCollection>  VertexCollection_out= std::auto_ptr<VertexCollection>(new VertexCollection);
  reco::VertexRefProd VertexRefProd_out = iEvent.getRefBeforePut<reco::VertexCollection>("PFTauPrimaryVertices");
  // For each Tau Run Algorithim 
  if(Tau.isValid()){
    unsigned int index(0);
    for(reco::PFTauCollection::size_type iPFTau = 0; iPFTau < Tau->size(); iPFTau++) {
      std::vector<reco::TrackBaseRef> SignalTracks;
      for(reco::PFTauCollection::size_type jPFTau = 0; jPFTau < Tau->size(); jPFTau++) {
	if(useSelectedTaus_ || iPFTau==jPFTau){
	  reco::PFTauRef RefPFTau(Tau, jPFTau);
	  ///////////////////////////////////////////////////////////////////////////////////////////////
	  // Get tracks form PFTau daugthers
	  const reco::PFCandidateRefVector & cands = RefPFTau->signalPFChargedHadrCands(); 
	  for (reco::PFCandidateRefVector::const_iterator iter = cands.begin(); iter!=cands.end(); iter++){
	    if(iter->get()->trackRef().isNonnull()) SignalTracks.push_back(reco::TrackBaseRef((*iter)->trackRef()));
	    else if(iter->get()->gsfTrackRef().isNonnull()){SignalTracks.push_back(reco::TrackBaseRef(((*iter)->gsfTrackRef())));}
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
      // Get Primary vertex
      reco::Vertex thePV;
      if(useInputPV==Algorithm_){
	if(Tau->size()==PV->size()){
	  thePV=PV->at(index);
	}
	else{
	  thePV=PV->front();
	  edm::LogError("PFTauPrimaryVertexProducer") <<"PFTauPrimaryVertexProducer Number of Tau do not match Number of Primary Vertices for useInputPV Algorithim. Using Tau Primary Vertes Instead";
	}
      }
      if(useFontPV==Algorithm_){
	thePV=PV->front();
      }
      ///////////////////////////////////////////////////////////////////////////////////////////////
      // Get Non-Tau tracks 
      reco::TrackCollection nonTauTracks;
      if (trackCollection.isValid()) {
	// remove tau tracks and only tracks associated with the vertex
	unsigned int idx = 0;
	for (reco::TrackCollection::const_iterator iTrk = trackCollection->begin(); iTrk != trackCollection->end(); ++iTrk, idx++) {
	  reco::TrackRef tmpRef(trackCollection, idx);
	  reco::TrackRef tmpRefForBase=tmpRef;
	  bool isSigTrk = false;
	  bool fromVertex=false;
	  for (unsigned int sigTrk = 0; sigTrk < SignalTracks.size(); sigTrk++) {
	    if (reco::TrackBaseRef(tmpRefForBase)==SignalTracks.at(sigTrk)){isSigTrk = true; break;}
	  }
	  for(std::vector<reco::TrackBaseRef>::const_iterator vtxTrkRef=thePV.tracks_begin();vtxTrkRef<thePV.tracks_end();vtxTrkRef++){
	    if(thePV.trackWeight(*vtxTrkRef)>0 ){
	      if((*vtxTrkRef)==reco::TrackBaseRef(tmpRefForBase)){fromVertex=true; break;}
	    }
	  }
	  if (!isSigTrk && fromVertex) nonTauTracks.push_back(*iTrk);
	}
      }
      ///////////////////////////////////////////////////////////////////////////////////////////////
      // Refit the vertex
      TransientVertex transVtx;
      std::vector<reco::TransientTrack> transTracks;
      for (reco::TrackCollection::iterator iter=nonTauTracks.begin(); iter!=nonTauTracks.end(); ++iter){
	transTracks.push_back(transTrackBuilder->build(*iter));
      }
      bool FitOk(true);
      AdaptiveVertexFitter avf;
      avf.setWeightThreshold(0.1); //weight per track. allow almost every fit, else --> exception
      try{
	if(!useBeamSpot_){transVtx = avf.vertex(transTracks);}
	else{transVtx = avf.vertex(transTracks,*beamSpot);}
      }catch(...){
	FitOk=false;
      }
      reco::Vertex primaryVertexReFit;
      if(FitOk)primaryVertexReFit=transVtx;
      VertexRef VRef = reco::VertexRef(VertexRefProd_out, VertexCollection_out->size());
      VertexCollection_out->push_back(primaryVertexReFit);
      AVPFTauPV->setValue(iPFTau, VRef);
    }
  }
  iEvent.put(VertexCollection_out,"PFTauPrimaryVertices");
  iEvent.put(AVPFTauPV);
}

DEFINE_FWK_MODULE(PFTauPrimaryVertexProducer);
