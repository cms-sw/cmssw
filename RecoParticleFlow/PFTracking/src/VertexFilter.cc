#include "RecoParticleFlow/PFTracking/interface/VertexFilter.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackExtra.h"
#include "DataFormats/TrackReco/interface/TrackExtraFwd.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "FWCore/ParameterSet/interface/InputTag.h"
#include "TrackingTools/TransientTrackingRecHit/interface/TransientTrackingRecHit.h"
#include "TrackingTools/PatternTools/interface/Trajectory.h"
#include "TrackingTools/PatternTools/interface/TrajTrackAssociation.h"
//
// class decleration
//




VertexFilter::VertexFilter(const edm::ParameterSet& iConfig):
  conf_(iConfig)
{
  produces<reco::TrackCollection>();
  produces<TrackingRecHitCollection>();
  produces<reco::TrackExtraCollection>();
  produces<std::vector<Trajectory> >();
  produces<TrajTrackAssociationCollection>();

}


VertexFilter::~VertexFilter()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called to produce the data  ------------
void
VertexFilter::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  using namespace edm;
  using namespace reco;
  using namespace std;
  
  auto_ptr<TrackCollection> selTracks(new TrackCollection);
  auto_ptr<TrackingRecHitCollection> selHits(new TrackingRecHitCollection);
  auto_ptr<TrackExtraCollection> selTrackExtras(new TrackExtraCollection);
  auto_ptr<vector<Trajectory> > outputTJ(new vector<Trajectory> );
  auto_ptr<TrajTrackAssociationCollection> trajTrackMap( new TrajTrackAssociationCollection() );
  
  
  TrackExtraRefProd rTrackExtras = iEvent.getRefBeforePut<TrackExtraCollection>();
  TrackingRecHitRefProd rHits = iEvent.getRefBeforePut<TrackingRecHitCollection>();



  Handle<std::vector<Trajectory> > TrajectoryCollection;
  Handle<TrajTrackAssociationCollection> assoMap;
  Handle<VertexCollection> vtxCollection;
  

  InputTag tkTag             = conf_.getParameter<InputTag>("recTracks");
  InputTag vtxTag            = conf_.getParameter<InputTag>("recVertices");
  
  int minhits = conf_.getParameter<int>("MinHits");
  float distz = conf_.getParameter<double>("DistZFromVertex");
  float distrho = conf_.getParameter<double>("DistRhoFromVertex");
  float chi_cut = conf_.getParameter<double>("ChiCut");

  Handle<TrackCollection> tkCollection;  
  iEvent.getByLabel( tkTag, tkCollection);
  TrackCollection  tC = *(tkCollection.product());
  TrackCollection::const_iterator itxc;
  


  
  
  iEvent.getByLabel(tkTag,TrajectoryCollection);
  iEvent.getByLabel(tkTag,assoMap);



  VertexCollection::const_iterator ivxc;
  iEvent.getByLabel( vtxTag            , vtxCollection);
  const VertexCollection  vxC = *(vtxCollection.product()); 

  float z_lead=0;
  if(vxC.size()==0){
    float pt_lead=0;
    float chi_lead=1000000;
    int nhit_lead=0;
    for(itxc=tC.begin();itxc!=tC.end();++itxc){
      if ((nhit_lead>6) && ((*itxc).found()<6))continue;
      
      if ((nhit_lead<6) && ((*itxc).found()>6)){
	z_lead=(*itxc).vertex().z();pt_lead=(*itxc).pt();
	chi_lead=(*itxc).normalizedChi2();nhit_lead=(*itxc).found();
	continue;
      }  
      if ((chi_lead<10) && ((*itxc).normalizedChi2()>10))	continue;
      if ((chi_lead>10) && ((*itxc).normalizedChi2()<10)){
	z_lead=(*itxc).vertex().z();pt_lead=(*itxc).pt();
	chi_lead=(*itxc).normalizedChi2();nhit_lead=(*itxc).found();
	continue;
      }
      
      if(pt_lead<(*itxc).pt()){
	z_lead=(*itxc).vertex().z();pt_lead=(*itxc).pt();
	chi_lead=(*itxc).normalizedChi2();nhit_lead=(*itxc).found(); 
      }
    }
  }
  
  for(TrajTrackAssociationCollection::const_iterator it = assoMap->begin();
      it != assoMap->end(); ++it){
    const Ref<vector<Trajectory> > traj = it->key;
    const reco::TrackRef itc = it->val;
    math::XYZPoint gp1=(*itc).vertex();
    bool hasAVertex=false;
 
    for (ivxc= vxC.begin();ivxc!=vxC.end();ivxc++){
      math::XYZPoint  gp2=(*ivxc).position();
      math::XYZVector dgp=gp2-gp1;
       
     if ((gp1.Rho()<distrho) &&
	  (fabs(dgp.Z())<distz)     &&
	  ((*itc).found()>=minhits) &&
	  ((*itc).chi2()<chi_cut)) hasAVertex=true;
    }
   


    if((vxC.size()==0)&&(tC.size()>0)){
      
      if ((gp1.Rho()<distrho) &&	
	  (fabs(gp1.Z()-z_lead)<distz)     &&
	  ((*itc).found()>=minhits) &&
	  ((*itc).chi2()<chi_cut)) hasAVertex=true;
      
    }
    
    if (hasAVertex){
      Track track =(*itc);
      //tracks and trajectories
      selTracks->push_back( track );
      outputTJ->push_back( *traj );
      //TRACKING HITS
      trackingRecHit_iterator irhit   =(*itc).recHitsBegin();
      trackingRecHit_iterator lasthit =(*itc).recHitsEnd();
      for (; irhit!=lasthit; ++irhit)
	selHits->push_back((*irhit)->clone() );
      
    
 
 
    }

  }

  //PUT TRACKING HITS IN THE EVENT
  OrphanHandle<TrackingRecHitCollection> theRecoHits = iEvent.put(selHits );
  
  //PUT TRACK EXTRA IN THE EVENT
  for ( unsigned index = 0; index < selTracks->size(); ++index ) { 

    unsigned hits=0;

    TrackExtra aTrackExtra(selTracks->at(index).outerPosition(),
			   selTracks->at(index).outerMomentum(),
			   selTracks->at(index).outerOk(),
			   selTracks->at(index).innerPosition(),
			   selTracks->at(index).innerMomentum(),
			   selTracks->at(index).innerOk(),
			   selTracks->at(index).outerStateCovariance(),
			   selTracks->at(index).outerDetId(),
			   selTracks->at(index).innerStateCovariance(),
			   selTracks->at(index).innerDetId(),
			   selTracks->at(index).seedDirection(),
			   selTracks->at(index).seedRef());

    unsigned nHits = selTracks->at(index).numberOfValidHits();
    for ( unsigned int ih=0; ih<nHits; ++ih) {
      aTrackExtra.add(TrackingRecHitRef(theRecoHits,hits++));
    }
    selTrackExtras->push_back(aTrackExtra);
  }

  //CORRECT REF TO TRACK
  OrphanHandle<TrackExtraCollection> theRecoTrackExtras = iEvent.put(selTrackExtras); 
  unsigned nTracks = selTracks->size();
  for ( unsigned index = 0; index<nTracks; ++index ) { 
    const reco::TrackExtraRef theTrackExtraRef(theRecoTrackExtras,index);
    (selTracks->at(index)).setExtra(theTrackExtraRef);
  }

  //TRACKS AND TRAJECTORIES
  OrphanHandle<TrackCollection> theRecoTracks = iEvent.put(selTracks);
  OrphanHandle<vector<Trajectory> > theRecoTrajectories = iEvent.put(outputTJ);

  //TRACKS<->TRAJECTORIES MAP 
  for ( unsigned index = 0; index < theRecoTracks->size(); ++index ) { 
    Ref<vector<Trajectory> > trajRef( theRecoTrajectories, index );
    Ref<TrackCollection>    tkRef( theRecoTracks, index );
    trajTrackMap->insert(trajRef,tkRef);
  }
  //MAP IN THE EVENT
  iEvent.put( trajTrackMap );
}

// ------------ method called once each job just before starting event loop  ------------
void 
VertexFilter::beginJob(const edm::EventSetup&)
{
}

// ------------ method called once each job just after ending the event loop  ------------
void 
VertexFilter::endJob() {
}


