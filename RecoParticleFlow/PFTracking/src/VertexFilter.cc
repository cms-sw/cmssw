#include "RecoParticleFlow/PFTracking/interface/VertexFilter.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackExtra.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "FWCore/ParameterSet/interface/InputTag.h"
#include "TrackingTools/TransientTrackingRecHit/interface/TransientTrackingRecHit.h"
#include "TrackingTools/PatternTools/interface/Trajectory.h"
#include "TrackingTools/PatternTools/interface/TrajTrackAssociation.h"
//
// class decleration
//


class LeadTrack{
 public:
  LeadTrack(){};
  bool operator()( reco::Track tk1,
		   reco::Track tk2){
    if((tk1.pt()>3)&&(tk2.pt()<3.)) return true;
    if((tk1.pt()<3)&&(tk2.pt()>3.)) return false;
    if((tk1.found()>6)&&(tk2.found()<6)) return true;
    if((tk1.found()<6)&&(tk2.found()>6)) return false;
    return tk1.normalizedChi2()<tk2.normalizedChi2();
  };
};

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
  
  Ref<TrackExtraCollection>::key_type idx = 0;
  Ref<TrackCollection>::key_type iTkRef = 0;
  
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
  stable_sort(tC.begin(),tC.end(),LeadTrack());




  iEvent.getByLabel(tkTag,TrajectoryCollection);
  iEvent.getByLabel(tkTag,assoMap);



  VertexCollection::const_iterator ivxc;
  iEvent.getByLabel( vtxTag            , vtxCollection);
  const VertexCollection  vxC = *(vtxCollection.product()); 


  
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
      math::XYZPoint  gp2=(*tC.begin()).vertex();
      math::XYZVector dgp=gp2-gp1;
      if ((gp1.Rho()<distrho) &&	
	  (fabs(dgp.Z())<distz)     &&
	  ((*itc).found()>=minhits) &&
	  ((*itc).chi2()<chi_cut)) hasAVertex=true;
      
    }
    
    if (hasAVertex){
      Track track =(*itc);

      
      
      iTkRef++;
      TrackExtraRef teref=TrackExtraRef ( rTrackExtras, idx ++ );
      track.setExtra( teref );
      TrackExtra & tx =const_cast<TrackExtra &>((*(*itc).extra()));
      
      
 
    trackingRecHit_iterator irhit=(*itc).recHitsBegin();
     size_t i = 0;
     for (; irhit!=(*itc).recHitsEnd(); irhit++){
	  TrackingRecHit *hit=(*(*irhit)).clone();
//	  track.setHitPattern( * hit, i ++ );
	  selHits->push_back(hit );
     }

      selTrackExtras->push_back(tx);
      outputTJ->push_back(*traj);	
      selTracks->push_back( track );
    }
  }
  
  
  
  iEvent.put(selTracks);
  iEvent.put(selTrackExtras);    
  iEvent.put(selHits);
  iEvent.put(outputTJ);
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


