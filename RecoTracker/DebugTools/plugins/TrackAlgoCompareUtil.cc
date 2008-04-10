#include "RecoTracker/DebugTools/plugins/TrackAlgoCompareUtil.h"

using namespace std;
using namespace edm;


// constructors and destructor
TrackAlgoCompareUtil::TrackAlgoCompareUtil(const edm::ParameterSet& iConfig)
{
  //now do what ever other initialization is needed
  trackLabel_algoA = iConfig.getParameter<edm::InputTag>("trackLabel_algoA");
  trackLabel_algoB = iConfig.getParameter<edm::InputTag>("trackLabel_algoB");
  trackingParticleLabel_fakes = iConfig.getParameter<edm::InputTag>("trackingParticleLabel_fakes");
  trackingParticleLabel_effic = iConfig.getParameter<edm::InputTag>("trackingParticleLabel_effic");
  vertexLabel_algoA = iConfig.getParameter<edm::InputTag>("vertexLabel_algoA");
  vertexLabel_algoB = iConfig.getParameter<edm::InputTag>("vertexLabel_algoB");
  trackingVertexLabel = iConfig.getParameter<edm::InputTag>("trackingVertexLabel");
  beamSpotLabel = iConfig.getParameter<edm::InputTag>("beamSpotLabel");
  assocByHitsLabel = iConfig.getUntrackedParameter<std::string>("assocByHitsLabel", "TrackAssociatorByHits");
  
  produces<TPtoRecoTrackCollection>("AlgoA");
  produces<TPtoRecoTrackCollection>("AlgoB");
  produces<TPtoRecoTrackCollection>("TP");
}


TrackAlgoCompareUtil::~TrackAlgoCompareUtil()
{
}


// ------------ method called once each job just before starting event loop  ------------
void TrackAlgoCompareUtil::beginJob(const edm::EventSetup&)
{
  numEventsProcessed = 0;
}


// ------------ method called to produce the data  ------------
void
TrackAlgoCompareUtil::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  cout << "Event Number = " << ++numEventsProcessed << endl;
  
  // create output collection instance
  std::auto_ptr<TPtoRecoTrackCollection> outputAlgoA(new TPtoRecoTrackCollection());
  std::auto_ptr<TPtoRecoTrackCollection> outputAlgoB(new TPtoRecoTrackCollection());
  std::auto_ptr<TPtoRecoTrackCollection> outputTP(new TPtoRecoTrackCollection());
  
  edm::Handle<reco::BeamSpot> recoBeamSpotHandle;
  iEvent.getByLabel(beamSpotLabel, recoBeamSpotHandle);
  reco::BeamSpot beamSpot = *recoBeamSpotHandle; 
  
  // Get Inputs
  edm::Handle<View<reco::Track> > trackCollAlgoA;
  //edm::Handle<reco::TrackCollection> trackCollAlgoA;
  iEvent.getByLabel(trackLabel_algoA, trackCollAlgoA);
  
  edm::Handle< View<reco::Track> > trackCollAlgoB;
  //edm::Handle<reco::TrackCollection> trackCollAlgoB;
  iEvent.getByLabel(trackLabel_algoB, trackCollAlgoB);
  
  edm::Handle<TrackingParticleCollection> trackingParticleCollFakes;
  iEvent.getByLabel(trackingParticleLabel_fakes, trackingParticleCollFakes);
  
  edm::Handle<TrackingParticleCollection> trackingParticleCollEffic;
  iEvent.getByLabel(trackingParticleLabel_effic, trackingParticleCollEffic);
  
  edm::Handle<reco::VertexCollection> vertexCollAlgoA;
  iEvent.getByLabel(vertexLabel_algoA, vertexCollAlgoA);
  
  edm::Handle<reco::VertexCollection> vertexCollAlgoB;
  iEvent.getByLabel(vertexLabel_algoB, vertexCollAlgoB);
  
  edm::Handle<TrackingVertexCollection> trackingVertexColl;
  iEvent.getByLabel(trackingVertexLabel, trackingVertexColl);
  
  // get associator (by hits) for the track
  edm::ESHandle<TrackAssociatorBase> theAssociatorByHits;
  iSetup.get<TrackAssociatorRecord>().get(assocByHitsLabel, theAssociatorByHits);
  
  // call the associator functions:
  reco::RecoToSimCollection recSimCollByHits_AlgoA = theAssociatorByHits->associateRecoToSim(trackCollAlgoA,trackingParticleCollFakes, &iEvent);
  reco::RecoToSimCollection recSimCollByHits_AlgoB = theAssociatorByHits->associateRecoToSim(trackCollAlgoB,trackingParticleCollFakes, &iEvent);
  
  reco::SimToRecoCollection simRecCollByHits_AlgoA = theAssociatorByHits->associateSimToReco(trackCollAlgoA,trackingParticleCollEffic, &iEvent);
  reco::SimToRecoCollection simRecCollByHits_AlgoB = theAssociatorByHits->associateSimToReco(trackCollAlgoB,trackingParticleCollEffic, &iEvent);
  
  // define the vector of references to trackingParticleColl associated with a given recoTrack
  std::vector<std::pair<TrackingParticleRef, double> > associatedTrackingParticles;
  
  // define the vector of references to trackColl associated with a given trackingParticle
  std::vector<std::pair<reco::TrackBaseRef, double> > associatedRecoTracks;
  
  // Get the magnetic field data from the event (used to calculate the point of closest approach for simTracks)
  edm::ESHandle<MagneticField> theMagneticField;
  iSetup.get<IdealMagneticFieldRecord>().get(theMagneticField);
  const MagneticField *magneticField = theMagneticField.product();
	
  
  // fill collection algoA
  for(View<reco::Track>::size_type i = 0; i < trackCollAlgoA->size(); ++i)
    //for(reco::TrackCollection::size_type i = 0; i < trackCollAlgoA->size(); ++i)
    {
      // get recoTrack algo A
      reco::TrackBaseRef recoTrack(trackCollAlgoA, i);
      TPtoRecoTrack  recoTracktoTP;
      recoTracktoTP.SetRecoTrack_AlgoA(recoTrack);
      recoTracktoTP.SetBeamSpot(beamSpot.position());
		
      // get the associated trackingParticle
      if(recSimCollByHits_AlgoA.find(recoTrack) != recSimCollByHits_AlgoA.end())
        {
	  associatedTrackingParticles = recSimCollByHits_AlgoA[recoTrack];
	  recoTracktoTP.SetTrackingParticle( associatedTrackingParticles.begin()->first );
	  SetTrackingParticleD0Dz(associatedTrackingParticles.begin()->first, beamSpot, magneticField, recoTracktoTP);
	  //SetTrackingParticleD0Dz(associatedTrackingParticles.begin()->first, recoTrack, magneticField, recoTracktoTP);
	}
      else
	{			
	  recoTracktoTP.SetTrackingParticle(TrackingParticleRef());
	}		
      // get the reco primary vertex info
      if(vertexCollAlgoA->size())  
	{
	  recoTracktoTP.SetRecoVertex_AlgoA( reco::VertexRef(vertexCollAlgoA, 0) );
	}
      else
	{
	  recoTracktoTP.SetRecoVertex_AlgoA( reco::VertexRef() );
	}
      // get the tracking (sim) primary vertex info
      if(trackingVertexColl->size())
        {
	  recoTracktoTP.SetTrackingVertex( TrackingVertexRef(trackingVertexColl, 0) );
        }
      else
        {
	  recoTracktoTP.SetTrackingVertex( TrackingVertexRef() );
        }
		
      outputAlgoA->push_back(recoTracktoTP);
    }
  
  
  // fill collection algoB
  for(reco::TrackCollection::size_type i = 0; i < trackCollAlgoB->size(); ++i)
    {
      // get recoTrack algo B
      reco::TrackBaseRef recoTrack(trackCollAlgoB, i);
      TPtoRecoTrack  recoTracktoTP;
      recoTracktoTP.SetRecoTrack_AlgoB(recoTrack);
      recoTracktoTP.SetBeamSpot(beamSpot.position());
      
      // get the associated trackingParticle
      if(recSimCollByHits_AlgoB.find(recoTrack) != recSimCollByHits_AlgoB.end())
        {
	  associatedTrackingParticles = recSimCollByHits_AlgoB[recoTrack];
	  recoTracktoTP.SetTrackingParticle( associatedTrackingParticles.begin()->first );
	  SetTrackingParticleD0Dz(associatedTrackingParticles.begin()->first, beamSpot, magneticField, recoTracktoTP);
	}
      else
	{			
	  recoTracktoTP.SetTrackingParticle(TrackingParticleRef());
	}		
      // get the reco primary vertex info
      if(vertexCollAlgoB->size())  
	{
	  recoTracktoTP.SetRecoVertex_AlgoB( reco::VertexRef(vertexCollAlgoB, 0) );
	}
      else
	{
	  recoTracktoTP.SetRecoVertex_AlgoB( reco::VertexRef() );
	}
      // get the tracking (sim) primary vertex info
      if(trackingVertexColl->size())
        {
	  recoTracktoTP.SetTrackingVertex( TrackingVertexRef(trackingVertexColl, 0) );
        }
        else
	  {
            recoTracktoTP.SetTrackingVertex( TrackingVertexRef() );
	  }

      outputAlgoB->push_back(recoTracktoTP);
    }
  
  
  for(TrackingParticleCollection::size_type i = 0; i < trackingParticleCollEffic->size(); ++i)
    {
      // initialize the trackingParticle (sim) info
      TrackingParticleRef tparticle(trackingParticleCollEffic, i);
      TPtoRecoTrack  tptoRecoTrack;
      tptoRecoTrack.SetBeamSpot(beamSpot.position());
      tptoRecoTrack.SetTrackingParticle(tparticle);
      SetTrackingParticleD0Dz(tparticle, beamSpot,  magneticField, tptoRecoTrack);
      
      // get the tracking (sim) primary vertex info
      if(trackingVertexColl->size())
        {
	  tptoRecoTrack.SetTrackingVertex( TrackingVertexRef(trackingVertexColl, 0) );
        }
      else
        {
	  tptoRecoTrack.SetTrackingVertex( TrackingVertexRef() );
        }
      
      
      // get the assocated recoTrack algoA
      if(simRecCollByHits_AlgoA.find(tparticle) != simRecCollByHits_AlgoA.end())
        {
	  associatedRecoTracks = simRecCollByHits_AlgoA[tparticle];
	  tptoRecoTrack.SetRecoTrack_AlgoA(associatedRecoTracks.begin()->first );
        }
      else
        {
	  tptoRecoTrack.SetRecoTrack_AlgoA(reco::TrackBaseRef());
        }
      // get the recoVertex algo A
      if(vertexCollAlgoA->size())
        {
	  tptoRecoTrack.SetRecoVertex_AlgoA( reco::VertexRef(vertexCollAlgoA, 0) );
        }
      else
	{
	  tptoRecoTrack.SetRecoVertex_AlgoA( reco::VertexRef() );
        }
      
      // get the assocated recoTrack algoB
      if(simRecCollByHits_AlgoB.find(tparticle) != simRecCollByHits_AlgoB.end())
        {
	  associatedRecoTracks = simRecCollByHits_AlgoB[tparticle];
	  tptoRecoTrack.SetRecoTrack_AlgoB(associatedRecoTracks.begin()->first );
        }
      else
        {
	  tptoRecoTrack.SetRecoTrack_AlgoB(reco::TrackBaseRef());
        }
      // get the recoVertex algo B
      if(vertexCollAlgoB->size())
        {
	  tptoRecoTrack.SetRecoVertex_AlgoB( reco::VertexRef(vertexCollAlgoB, 0) );
        }
      else
        {
	  tptoRecoTrack.SetRecoVertex_AlgoB( reco::VertexRef() );
        }
      
      outputTP->push_back(tptoRecoTrack);
    }


  // put the collection in the event record
  iEvent.put(outputAlgoA, "AlgoA");
  iEvent.put(outputAlgoB, "AlgoB");
  iEvent.put(outputTP, "TP");
}

// ------------ method called once each job just after ending the event loop  ------------
void TrackAlgoCompareUtil::endJob() 
{
}


// ------------ Producer Specific Meber Fucntions ----------------------------------------
void TrackAlgoCompareUtil::SetTrackingParticleD0Dz(TrackingParticleRef tp, const reco::BeamSpot &bs, const MagneticField *bf, TPtoRecoTrack& TPRT)
{
  GlobalPoint trackingParticleVertex( tp->vertex().x(), tp->vertex().y(), tp->vertex().z() );
  GlobalVector trackingParticleP3(tp->g4Track_begin()->momentum().x(),
				  tp->g4Track_begin()->momentum().y(),
				  tp->g4Track_begin()->momentum().z() );
  TrackCharge trackingParticleCharge(tp->charge());
  
  FreeTrajectoryState ftsAtProduction( trackingParticleVertex, trackingParticleP3, trackingParticleCharge, bf );
  TrajectoryStateClosestToBeamLineBuilder tscblBuilder;
  TrajectoryStateClosestToBeamLine tsAtClosestApproach = tscblBuilder(ftsAtProduction, bs);//as in TrackProducerAlgorithm
  
	if(tsAtClosestApproach.isValid())
	  {
	    GlobalPoint v1 = tsAtClosestApproach.trackStateAtPCA().position();
	    GlobalVector p = tsAtClosestApproach.trackStateAtPCA().momentum();

	    TPRT.SetTrackingParticleMomentumPCA(p);
	    TPRT.SetTrackingParticlePCA(v1);
	  }
	else
	  {
	    TPRT.SetTrackingParticleMomentumPCA(GlobalVector(-9999.0, -9999.0, -9999.0));
	    TPRT.SetTrackingParticlePCA(GlobalPoint(-9999.0, -9999.0, -9999.0));
	}
	
	
}

