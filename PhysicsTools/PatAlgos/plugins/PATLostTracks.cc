#include <string>


#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidateFwd.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "DataFormats/PatCandidates/interface/PackedCandidate.h"
#include "DataFormats/PatCandidates/interface/Jet.h"
#include "DataFormats/Common/interface/Association.h"
#include "FWCore/Framework/interface/global/EDProducer.h"
#include "DataFormats/Common/interface/View.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrack.h"
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "DataFormats/PatCandidates/interface/Jet.h"
#include "DataFormats/Common/interface/Association.h"
#include "DataFormats/RecoCandidate/interface/RecoChargedCandidate.h"

namespace {
  bool passesQuality(const reco::Track& trk,const std::vector<reco::TrackBase::TrackQuality>& allowedQuals){
    for(const auto& qual : allowedQuals){
      if(trk.quality(qual)) return true;
    }
    return false;
  }
}

namespace pat {
  class PATLostTracks : public edm::global::EDProducer<> {
  public:
    explicit PATLostTracks(const edm::ParameterSet&);
    ~PATLostTracks() override;
    
    void produce(edm::StreamID, edm::Event&, const edm::EventSetup&) const override;
   
  private:  
    enum class TrkStatus {
      NOTUSED=0,
      PFCAND,
      PFCANDNOTRKPROPS,
      PFELECTRON,
      PFPOSITRON,
      VTX
    };
    bool passTrkCuts(const reco::Track& tr)const;
    void addPackedCandidate(std::vector<pat::PackedCandidate>& cands,
			    const reco::TrackRef& trk,
			    const reco::VertexRef& pvSlimmed,
			    const reco::VertexRefProd& pvSlimmedColl,
			    const reco::Vertex& pvOrig,
			    const TrkStatus trkStatus)const;
      
  private:
    const edm::EDGetTokenT<reco::PFCandidateCollection>    cands_;
    const edm::EDGetTokenT<edm::Association<pat::PackedCandidateCollection> > map_;
    const edm::EDGetTokenT<reco::TrackCollection>          tracks_;
    const edm::EDGetTokenT<reco::VertexCollection>         vertices_;
    const edm::EDGetTokenT<reco::VertexCompositeCandidateCollection>         kshorts_;
    const edm::EDGetTokenT<reco::VertexCompositeCandidateCollection>         lambdas_;
    const edm::EDGetTokenT<reco::VertexCollection>         pv_;
    const edm::EDGetTokenT<reco::VertexCollection>         pvOrigs_;
    const double minPt_;
    const double minHits_;
    const double minPixelHits_;
    const double minPtToStoreProps_;
    const int covarianceVersion_;
    const int covarianceSchema_;
    std::vector<reco::TrackBase::TrackQuality> qualsToAutoAccept_;
  };
}

pat::PATLostTracks::PATLostTracks(const edm::ParameterSet& iConfig) :
  cands_(consumes<reco::PFCandidateCollection>(iConfig.getParameter<edm::InputTag>("inputCandidates"))),
  map_(consumes<edm::Association<pat::PackedCandidateCollection> >(iConfig.getParameter<edm::InputTag>("packedPFCandidates"))),
  tracks_(consumes<reco::TrackCollection>(iConfig.getParameter<edm::InputTag>("inputTracks"))),
  vertices_(consumes<reco::VertexCollection>(iConfig.getParameter<edm::InputTag>("secondaryVertices"))),
  kshorts_(consumes<reco::VertexCompositeCandidateCollection>(iConfig.getParameter<edm::InputTag>("kshorts"))),
  lambdas_(consumes<reco::VertexCompositeCandidateCollection>(iConfig.getParameter<edm::InputTag>("lambdas"))),
  pv_(consumes<reco::VertexCollection>(iConfig.getParameter<edm::InputTag>("primaryVertices"))),
  pvOrigs_(consumes<reco::VertexCollection>(iConfig.getParameter<edm::InputTag>("originalVertices"))),
  minPt_(iConfig.getParameter<double>("minPt")),
  minHits_(iConfig.getParameter<uint32_t>("minHits")),
  minPixelHits_(iConfig.getParameter<uint32_t>("minPixelHits")) ,
  minPtToStoreProps_(iConfig.getParameter<double>("minPtToStoreProps")),
  covarianceVersion_(iConfig.getParameter<int >("covarianceVersion")),
  covarianceSchema_(iConfig.getParameter<int >("covarianceSchema"))

{ 
  std::vector<std::string> trkQuals(iConfig.getParameter<std::vector<std::string> >("qualsToAutoAccept"));
  std::transform(trkQuals.begin(),trkQuals.end(),std::back_inserter(qualsToAutoAccept_),reco::TrackBase::qualityByName);
  
  if(std::find(qualsToAutoAccept_.begin(),qualsToAutoAccept_.end(),reco::TrackBase::undefQuality)!=qualsToAutoAccept_.end()){
    std::ostringstream msg;
    msg<<" PATLostTracks has a quality requirement which resolves to undefQuality. This usually means a typo and is therefore treated a config error\nquality requirements:\n   ";
    for(const auto& trkQual : trkQuals) msg <<trkQual<<" ";
    throw cms::Exception("Configuration") << msg.str();
  } 
    
  produces< std::vector<reco::Track> > ();
  produces< std::vector<pat::PackedCandidate> > (); produces< std::vector<pat::PackedCandidate> > ("eleTracks");
  produces< edm::Association<pat::PackedCandidateCollection> > ();
}

pat::PATLostTracks::~PATLostTracks() {}

void pat::PATLostTracks::produce(edm::StreamID, edm::Event& iEvent, const edm::EventSetup& iSetup) const {

    edm::Handle<reco::PFCandidateCollection> cands;
    iEvent.getByToken( cands_, cands );

    edm::Handle<edm::Association<pat::PackedCandidateCollection> > pf2pc;
    iEvent.getByToken(map_,pf2pc);

    edm::Handle<reco::TrackCollection> tracks;
    iEvent.getByToken( tracks_, tracks );
    
    edm::Handle<reco::VertexCollection> vertices;
    iEvent.getByToken( vertices_, vertices );

    edm::Handle<reco::VertexCompositeCandidateCollection> kshorts;
    iEvent.getByToken( kshorts_, kshorts );
    edm::Handle<reco::VertexCompositeCandidateCollection> lambdas;
    iEvent.getByToken( lambdas_, lambdas );

    edm::Handle<reco::VertexCollection> pvs;
    iEvent.getByToken( pv_, pvs );
    reco::VertexRef pv(pvs.id());
    reco::VertexRefProd pvRefProd(pvs);
    if (!pvs->empty()) {
        pv = reco::VertexRef(pvs, 0);
    }
    edm::Handle<reco::VertexCollection> pvOrigs;
    iEvent.getByToken( pvOrigs_, pvOrigs );
    const reco::Vertex & pvOrig = (*pvOrigs)[0];

    auto outPtrTrks = std::make_unique<std::vector<reco::Track>>();
    auto outPtrTrksAsCands = std::make_unique<std::vector<pat::PackedCandidate>>();
    auto outPtrEleTrksAsCands = std::make_unique<std::vector<pat::PackedCandidate>>();
  
    std::vector<TrkStatus> trkStatus(tracks->size(),TrkStatus::NOTUSED);
    //Mark all tracks used in candidates	
    //check if packed candidates are storing the tracks by seeing if number of hits >0 	  
    //currently we dont use that information though
    //electrons will never store their track (they store the GSF track)
    for(unsigned int ic=0, nc = cands->size(); ic < nc; ++ic) {
        edm::Ref<reco::PFCandidateCollection> r(cands,ic);
        const reco::PFCandidate &cand=(*cands)[ic]; 
	if(cand.charge() && cand.trackRef().isNonnull() && cand.trackRef().id() == tracks.id() ) { 
	  
	  if(cand.pdgId()==11) trkStatus[cand.trackRef().key()]=TrkStatus::PFELECTRON;	  
	  else if(cand.pdgId()==-11) trkStatus[cand.trackRef().key()]=TrkStatus::PFPOSITRON;
	  else if((*pf2pc)[r]->numberOfHits() > 0) trkStatus[cand.trackRef().key()]=TrkStatus::PFCAND; 
	  else trkStatus[cand.trackRef().key()]=TrkStatus::PFCANDNOTRKPROPS; 
	}
    }
        
    //Mark all tracks used in secondary vertices
    for(const auto& secVert : *vertices){
        for(auto trkIt = secVert.tracks_begin();trkIt!=secVert.tracks_end();trkIt++){
	    if(trkStatus[trkIt->key()]==TrkStatus::NOTUSED)  trkStatus[trkIt->key()]=TrkStatus::VTX;
	}
    }
    for(const auto& v0 : *kshorts){
        for(size_t dIdx=0;dIdx<v0.numberOfDaughters(); dIdx++){
	    size_t key= (dynamic_cast<const reco::RecoChargedCandidate*>(v0.daughter(dIdx)))->track().key();
	    if(trkStatus[key]==TrkStatus::NOTUSED)  trkStatus[key]=TrkStatus::VTX;
	}
    }
    for(const auto& v0 : *lambdas){
        for(size_t dIdx=0;dIdx<v0.numberOfDaughters(); dIdx++){
	    size_t key= (dynamic_cast<const reco::RecoChargedCandidate*>(v0.daughter(dIdx)))->track().key();
	    if(trkStatus[key]==TrkStatus::NOTUSED)  trkStatus[key]=TrkStatus::VTX;
	}
    }
    std::vector<int> mapping(tracks->size(),-1);  
    int lostTrkIndx=0;
    for(unsigned int trkIndx=0; trkIndx < tracks->size(); trkIndx++){
        reco::TrackRef trk(tracks,trkIndx);
	if( trkStatus[trkIndx] == TrkStatus::VTX || 
	   (trkStatus[trkIndx]==TrkStatus::NOTUSED && passTrkCuts(*trk)) ) { 
	
	    outPtrTrks->emplace_back(*trk);
	    addPackedCandidate(*outPtrTrksAsCands,trk,pv,pvRefProd,pvOrig,trkStatus[trkIndx]);
	
	    //for creating the reco::Track -> pat::PackedCandidate map
	    //not done for the lostTrack:eleTracks collection
	    mapping[trkIndx]=lostTrkIndx;
	    lostTrkIndx++;
	}else if( (trkStatus[trkIndx]==TrkStatus::PFELECTRON || trkStatus[trkIndx]==TrkStatus::PFPOSITRON ) 
		  && passTrkCuts(*trk) ) {
   	    addPackedCandidate(*outPtrEleTrksAsCands,trk,pv,pvRefProd,pvOrig,trkStatus[trkIndx]);

	
      }	      
    } 
    
    iEvent.put(std::move(outPtrTrks));
    iEvent.put(std::move(outPtrEleTrksAsCands),"eleTracks");
    edm::OrphanHandle<pat::PackedCandidateCollection> oh = iEvent.put(std::move(outPtrTrksAsCands));
    auto tk2pc = std::make_unique<edm::Association<pat::PackedCandidateCollection>>(oh);
    edm::Association<pat::PackedCandidateCollection>::Filler tk2pcFiller(*tk2pc);
    tk2pcFiller.insert(tracks, mapping.begin(), mapping.end());
    tk2pcFiller.fill() ; 
    iEvent.put(std::move(tk2pc));   
}

bool pat::PATLostTracks::passTrkCuts(const reco::Track& tr)const
{
    const bool passTrkHits = tr.pt() > minPt_ && 
                             tr.numberOfValidHits() >= minHits_ && 
                             tr.hitPattern().numberOfValidPixelHits() >= minPixelHits_;
    const bool passTrkQual = passesQuality(tr,qualsToAutoAccept_);

    return passTrkHits || passTrkQual;
}

void pat::PATLostTracks::addPackedCandidate(std::vector<pat::PackedCandidate>& cands,
					    const reco::TrackRef& trk,
					    const reco::VertexRef& pvSlimmed,
					    const reco::VertexRefProd& pvSlimmedColl,
					    const reco::Vertex& pvOrig,
					    const pat::PATLostTracks::TrkStatus trkStatus)const
{
    const float mass = 0.13957018;
    
    int id=211*trk->charge();
    if(trkStatus==TrkStatus::PFELECTRON) id=11;
    else if(trkStatus==TrkStatus::PFPOSITRON) id=-11; 
    
    reco::Candidate::PolarLorentzVector p4(trk->pt(),trk->eta(),trk->phi(),mass);
    cands.emplace_back(pat::PackedCandidate(p4,trk->vertex(),
					    trk->pt(),trk->eta(),trk->phi(),
					    id,pvSlimmedColl,pvSlimmed.key()));

    if(trk->pt()>minPtToStoreProps_ || trkStatus==TrkStatus::VTX) cands.back().setTrackProperties(*trk,covarianceSchema_,covarianceVersion_);
    if(pvOrig.trackWeight(trk) > 0.5) {
         cands.back().setAssociationQuality(pat::PackedCandidate::UsedInFitTight);
    }
}



using pat::PATLostTracks;
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(PATLostTracks);
