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




VertexFilter::VertexFilter(const edm::ParameterSet& iConfig)
{

  produces<reco::TrackCollection>();
  produces<TrackingRecHitCollection>();
  produces<reco::TrackExtraCollection>();
  produces<std::vector<Trajectory> >();
  produces<TrajTrackAssociationCollection>();

  tkTag  = iConfig.getParameter<edm::InputTag>("recTracks");
  vtxTag = iConfig.getParameter<edm::InputTag>("recVertices");
  
  minhits = iConfig.getParameter<int>("MinHits");
  distz = iConfig.getParameter<double>("DistZFromVertex");
  distrho = iConfig.getParameter<double>("DistRhoFromVertex");
  chi_cut = iConfig.getParameter<double>("ChiCut");

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
  
  Handle<TrackCollection> tkCollection;  
  iEvent.getByLabel( tkTag, tkCollection);
  const reco::TrackCollection*  tC = tkCollection.product();
  TrackCollection::const_iterator itxc;
  TrackCollection::const_iterator firstTrack = tC->begin();
  TrackCollection::const_iterator lastTrack = tC->end();
  unsigned tCsize = tC->size();
    
  iEvent.getByLabel(tkTag,TrajectoryCollection);
  iEvent.getByLabel(tkTag,assoMap);

  iEvent.getByLabel( vtxTag            , vtxCollection);
  const reco::VertexCollection*  vxC = vtxCollection.product(); 
  VertexCollection::const_iterator ivxc;
  VertexCollection::const_iterator firstVertex = vxC->begin();
  VertexCollection::const_iterator lastVertex = vxC->end();
  unsigned vxCsize = vxC->size();

  float z_lead=0.;
  unsigned sizeMax = 0;
  if(vxCsize==0){
    float pt_lead=0.;
    float chi_lead=1000000.;
    unsigned nhit_lead=0;
    for( itxc=firstTrack; itxc!=lastTrack;++itxc){
      unsigned foundHits = itxc->recHitsSize();
      sizeMax += foundHits;
      if ( nhit_lead>6 && foundHits<6 )continue;
      
      if ( nhit_lead<6 && foundHits>6 ){
	z_lead=itxc->vertex().z();
	pt_lead=itxc->pt();
	chi_lead=itxc->normalizedChi2();
	nhit_lead=foundHits;
	continue;
      }  
      if ( chi_lead<10. && itxc->normalizedChi2()>10. )	continue;
      if ( chi_lead>10. && itxc->normalizedChi2()<10. ){
	z_lead=itxc->vertex().z();
	pt_lead=itxc->pt();
	chi_lead=itxc->normalizedChi2();
	nhit_lead=foundHits;
	continue;
      }
      
      if(pt_lead<itxc->pt()){
	z_lead=itxc->vertex().z();
	pt_lead=itxc->pt();
	chi_lead=itxc->normalizedChi2();
	nhit_lead=foundHits; 
      }
    }
  }
  
  selHits->reserve(sizeMax);
  TrajTrackAssociationCollection::const_iterator it = assoMap->begin();
  TrajTrackAssociationCollection::const_iterator lastAssoc = assoMap->end();
  for( ; it != lastAssoc; ++it ) {
    const Ref<vector<Trajectory> > traj = it->key;
    const reco::TrackRef itc = it->val;
    math::XYZPoint gp1=itc->vertex();
    bool hasAVertex=false;
    unsigned foundHits = itc->recHitsSize();
 
    for (ivxc=firstVertex ;ivxc!=lastVertex;ivxc++){
      math::XYZPoint gp2=ivxc->position();
      math::XYZVector dgp=gp2-gp1;
       
      if ( gp1.Rho()<distrho &&
	   fabs(dgp.Z())<distz    &&
	   foundHits>=minhits &&
	   itc->chi2()<chi_cut ) hasAVertex=true;
    }
   


    if( vxCsize==0 && tCsize>0 ){
      
      if ( gp1.Rho()<distrho  &&	
	   fabs(gp1.Z()-z_lead)<distz     &&
	   foundHits>=minhits &&
	   itc->chi2()<chi_cut ) hasAVertex=true;
      
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
  unsigned nTracks = selTracks->size();
  selTrackExtras->reserve(nTracks);
  for ( unsigned index = 0; index<nTracks; ++index ) { 

    unsigned hits=0;

    reco::Track& aTrack = selTracks->at(index);
    TrackExtra aTrackExtra(aTrack.outerPosition(),
			   aTrack.outerMomentum(),
			   aTrack.outerOk(),
			   aTrack.innerPosition(),
			   aTrack.innerMomentum(),
			   aTrack.innerOk(),
			   aTrack.outerStateCovariance(),
			   aTrack.outerDetId(),
			   aTrack.innerStateCovariance(),
			   aTrack.innerDetId(),
			   aTrack.seedDirection(),
			   aTrack.seedRef());

    //unsigned nHits = aTrack.numberOfValidHits();
    unsigned nHits = aTrack.recHitsSize();
    for ( unsigned int ih=0; ih<nHits; ++ih) {
      aTrackExtra.add(TrackingRecHitRef(theRecoHits,hits++));
    }
    selTrackExtras->push_back(aTrackExtra);
  }

  //CORRECT REF TO TRACK
  OrphanHandle<TrackExtraCollection> theRecoTrackExtras = iEvent.put(selTrackExtras); 
  for ( unsigned index = 0; index<nTracks; ++index ) { 
    const reco::TrackExtraRef theTrackExtraRef(theRecoTrackExtras,index);
    (selTracks->at(index)).setExtra(theTrackExtraRef);
  }

  //TRACKS AND TRAJECTORIES
  OrphanHandle<TrackCollection> theRecoTracks = iEvent.put(selTracks);
  OrphanHandle<vector<Trajectory> > theRecoTrajectories = iEvent.put(outputTJ);

  //TRACKS<->TRAJECTORIES MAP 
  nTracks = theRecoTracks->size();
  for ( unsigned index = 0; index<nTracks; ++index ) { 
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


