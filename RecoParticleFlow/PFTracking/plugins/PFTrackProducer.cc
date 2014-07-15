#include <memory>
#include "RecoParticleFlow/PFTracking/interface/PFTrackProducer.h"
#include "RecoParticleFlow/PFTracking/interface/PFTrackTransformer.h"
#include "DataFormats/ParticleFlowReco/interface/PFRecTrack.h"
#include "DataFormats/ParticleFlowReco/interface/PFRecTrackFwd.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrack.h"
#include "DataFormats/EgammaReco/interface/ElectronSeed.h"
#include "DataFormats/EgammaReco/interface/ElectronSeedFwd.h"
#include "TrackingTools/PatternTools/interface/Trajectory.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "DataFormats/MuonReco/interface/Muon.h"
#include "TrackingTools/TransientTrack/interface/TransientTrackBuilder.h"
#include "TrackingTools/Records/interface/TransientTrackRecord.h"
#include "TrackingTools/IPTools/interface/IPTools.h"
#include "DataFormats/VertexReco/interface/Vertex.h"

using namespace std;
using namespace edm;
using namespace reco;
PFTrackProducer::PFTrackProducer(const ParameterSet& iConfig):
  pfTransformer_(0)
{
  produces<reco::PFRecTrackCollection>();
  
  
  std::vector<InputTag> tags=iConfig.getParameter< vector < InputTag > >("TkColList");
  for( unsigned int i=0;i<tags.size();++i)
    tracksContainers_.push_back(consumes<reco::TrackCollection>(tags[i]));

  useQuality_   = iConfig.getParameter<bool>("UseQuality");
  
  gsfTrackLabel_ = consumes<reco::GsfTrackCollection>(iConfig.getParameter<InputTag>
						      ("GsfTrackModuleLabel"));  

  trackQuality_=reco::TrackBase::qualityByName(iConfig.getParameter<std::string>("TrackQuality"));
  
  muonColl_ = consumes<reco::MuonCollection>(iConfig.getParameter< InputTag >("MuColl"));
  
  trajinev_ = iConfig.getParameter<bool>("TrajInEvents");
  
  gsfinev_ = iConfig.getParameter<bool>("GsfTracksInEvents");
  vtx_h=consumes<reco::VertexCollection>(iConfig.getParameter<edm::InputTag>("PrimaryVertexLabel"));
  
}

PFTrackProducer::~PFTrackProducer()
{
  delete pfTransformer_;
}

void
PFTrackProducer::produce(Event& iEvent, const EventSetup& iSetup)
{
  
  //create the empty collections 
  auto_ptr< reco::PFRecTrackCollection > 
    PfTrColl (new reco::PFRecTrackCollection);
  
  //read track collection
  Handle<GsfTrackCollection> gsftrackcoll;
  bool foundgsf = iEvent.getByToken(gsfTrackLabel_,gsftrackcoll);
  GsfTrackCollection gsftracks;
  //Get PV for STIP calculation, if there is none then take the dummy  
  Handle<reco::VertexCollection> vertex;
  iEvent.getByToken(vtx_h, vertex);
  reco::Vertex dummy;
  const reco::Vertex* pv=&dummy;  
  if (vertex.isValid()) 
    {
      pv = &*vertex->begin();    
    } 
  else 
    { // create a dummy PV
      reco::Vertex::Error e;
      e(0, 0) = 0.0015 * 0.0015;
      e(1, 1) = 0.0015 * 0.0015;
      e(2, 2) = 15. * 15.;
      reco::Vertex::Point p(0, 0, 0);
      dummy = reco::Vertex(p, e, 0, 0, 0);   
    } 
  
  edm::ESHandle<TransientTrackBuilder> builder;
  iSetup.get<TransientTrackRecord>().get("TransientTrackBuilder", builder);
  TransientTrackBuilder thebuilder = *(builder.product());
  
  
  if(gsfinev_) {
    if(foundgsf )
      gsftracks  = *(gsftrackcoll.product());
  }  
  
  // read muon collection
  Handle< reco::MuonCollection > recMuons;
  iEvent.getByToken(muonColl_, recMuons);
  
  
  for (unsigned int istr=0; istr<tracksContainers_.size();istr++){
    
    //Track collection
    Handle<reco::TrackCollection> tkRefCollection;
    iEvent.getByToken(tracksContainers_[istr], tkRefCollection);
    reco::TrackCollection  Tk=*(tkRefCollection.product());
    
    vector<Trajectory> Tj(0);
    if(trajinev_) {
      //Trajectory collection
      Handle<vector<Trajectory> > tjCollection;
      iEvent.getByToken(tracksContainers_[istr], tjCollection);
	
	Tj =*(tjCollection.product());
    }
    
    
    for(unsigned int i=0;i<Tk.size();i++){
      
      reco::TrackRef trackRef(tkRefCollection, i);
	
      if (useQuality_ &&
	  (!(Tk[i].quality(trackQuality_)))){
	
	bool isMuCandidate = false;
	
	//TrackRef trackRef(tkRefCollection, i);
	
	if(recMuons.isValid() ) {
	  for(unsigned j=0;j<recMuons->size(); j++) {
	    reco::MuonRef muonref( recMuons, j );
	    if (muonref->track().isNonnull()) 
	      if( muonref->track() == trackRef && muonref->isGlobalMuon()){
		isMuCandidate=true;
		//cout<<" SAVING TRACK "<<endl;
		break;
	      }
	  }
	}
	if(!isMuCandidate)
	  {
	    continue;	  
	  }
	
      }
           
      // find the pre-id kf track
      bool preId = false;
      if(foundgsf) {
	for (unsigned int igsf=0; igsf<gsftracks.size();igsf++) {
	  GsfTrackRef gsfTrackRef(gsftrackcoll, igsf);
	  if (gsfTrackRef->seedRef().isNull()) continue;
	  ElectronSeedRef ElSeedRef= gsfTrackRef->extra()->seedRef().castTo<ElectronSeedRef>();
	  if (ElSeedRef->ctfTrack().isNonnull()) {
	    if(ElSeedRef->ctfTrack() == trackRef) preId = true;
	  }
	}
      }
      if(preId) {
	// Set PFRecTrack of type KF_ElCAND
	reco::PFRecTrack pftrack( trackRef->charge(), 
				  reco::PFRecTrack::KF_ELCAND, 
				  i, trackRef );
	
	bool valid = false;
	if(trajinev_) {
	  valid = pfTransformer_->addPoints( pftrack, *trackRef, Tj[i]);
	}
	else {
	  Trajectory FakeTraj;
	  valid = pfTransformer_->addPoints( pftrack, *trackRef, FakeTraj);
	}
	if(valid) {
	  //calculate STIP
	  double stip=-999;
	  const reco::PFTrajectoryPoint& atECAL=pftrack.extrapolatedPoint(reco::PFTrajectoryPoint::ECALEntrance);
	  if(atECAL.isValid()) //if track extrapolates to ECAL
	    {
	      GlobalVector direction(pftrack.extrapolatedPoint(reco::PFTrajectoryPoint::ECALEntrance).position().x(),
				     pftrack.extrapolatedPoint(reco::PFTrajectoryPoint::ECALEntrance).position().y(), 
				     pftrack.extrapolatedPoint(reco::PFTrajectoryPoint::ECALEntrance).position().z());
	      stip = IPTools::signedTransverseImpactParameter(thebuilder.build(*trackRef), direction, *pv).second.significance();
	    }
	  pftrack.setSTIP(stip);
	  PfTrColl->push_back(pftrack);
	}		
      }
      else {
	reco::PFRecTrack pftrack( trackRef->charge(), 
				  reco::PFRecTrack::KF, 
				  i, trackRef );
	bool valid = false;
	if(trajinev_) {
	  valid = pfTransformer_->addPoints( pftrack, *trackRef, Tj[i]);
	}
	else {
	  Trajectory FakeTraj;
	  valid = pfTransformer_->addPoints( pftrack, *trackRef, FakeTraj);
	}
	
	if(valid) {
	  double stip=-999;
	  const reco::PFTrajectoryPoint& atECAL=pftrack.extrapolatedPoint(reco::PFTrajectoryPoint::ECALEntrance);
	  if(atECAL.isValid())
	    {
	      GlobalVector direction(pftrack.extrapolatedPoint(reco::PFTrajectoryPoint::ECALEntrance).position().x(),
				     pftrack.extrapolatedPoint(reco::PFTrajectoryPoint::ECALEntrance).position().y(), 
				     pftrack.extrapolatedPoint(reco::PFTrajectoryPoint::ECALEntrance).position().z());
	      stip = IPTools::signedTransverseImpactParameter(thebuilder.build(*trackRef), direction, *pv).second.significance();
	    }
	  pftrack.setSTIP(stip);
	  PfTrColl->push_back(pftrack);
	}
      }
    }
  }
  iEvent.put(PfTrColl);
}

// ------------ method called once each job just before starting event loop  ------------
void 
PFTrackProducer::beginRun(const edm::Run& run,
			  const EventSetup& iSetup)
{
  ESHandle<MagneticField> magneticField;
  iSetup.get<IdealMagneticFieldRecord>().get(magneticField);
  pfTransformer_= new PFTrackTransformer(math::XYZVector(magneticField->inTesla(GlobalPoint(0,0,0))));
  if(!trajinev_)
    pfTransformer_->OnlyProp();
}

// ------------ method called once each job just after ending the event loop  ------------
void 
PFTrackProducer::endRun(const edm::Run& run,
			const EventSetup& iSetup) {
  delete pfTransformer_;
  pfTransformer_=nullptr;
}
