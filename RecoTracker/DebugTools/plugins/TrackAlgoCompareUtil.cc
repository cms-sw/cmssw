#include "RecoTracker/DebugTools/plugins/TrackAlgoCompareUtil.h"

using namespace std;
using namespace edm;


// constructors and destructor
TrackAlgoCompareUtil::TrackAlgoCompareUtil(const edm::ParameterSet& iConfig):
  trackLabel_algoA(consumes<View<reco::Track>>(iConfig.getParameter<edm::InputTag>("trackLabel_algoA"))),
  trackLabel_algoB(consumes<View<reco::Track>>(iConfig.getParameter<edm::InputTag>("trackLabel_algoB"))),
  trackingParticleLabel_fakes(consumes<TrackingParticleCollection>(iConfig.getParameter<edm::InputTag>("trackingParticleLabel_fakes"))),
  trackingParticleLabel_effic(consumes<TrackingParticleCollection>(iConfig.getParameter<edm::InputTag>("trackingParticleLabel_effic"))),
  beamSpotLabel(consumes<reco::BeamSpot>(iConfig.getParameter<edm::InputTag>("beamSpotLabel"))),
  UseAssociators(iConfig.getParameter< bool >("UseAssociators")),
  UseVertex(iConfig.getParameter< bool >("UseVertex"))
{
    //now do what ever other initialization is needed
  if(UseVertex) {
    vertexLabel_algoA = consumes<reco::VertexCollection>(iConfig.getParameter<edm::InputTag>("vertexLabel_algoA"));
    vertexLabel_algoB = consumes<reco::VertexCollection>(iConfig.getParameter<edm::InputTag>("vertexLabel_algoB"));
  }

  if(UseAssociators) {
    assocLabel_algoA = consumes<reco::TrackToTrackingParticleAssociator>(iConfig.getUntrackedParameter<std::string>("assocLabel_algoA", "trackAssociatorByHits"));
    assocLabel_algoB = consumes<reco::TrackToTrackingParticleAssociator>(iConfig.getUntrackedParameter<std::string>("assocLabel_algoB", "trackAssociatorByHits"));
  }
  else {
    edm::InputTag algoA = iConfig.getParameter< edm::InputTag >("associatormap_algoA");
    edm::InputTag algoB = iConfig.getParameter< edm::InputTag >("associatormap_algoB");

    associatormap_algoA_recoToSim = consumes<reco::RecoToSimCollection>(algoA);
    associatormap_algoB_recoToSim = consumes<reco::RecoToSimCollection>(algoB);
    associatormap_algoA_simToReco = consumes<reco::SimToRecoCollection>(algoA);
    associatormap_algoB_simToReco = consumes<reco::SimToRecoCollection>(algoB);
  }
  
    produces<RecoTracktoTPCollection>("AlgoA");
    produces<RecoTracktoTPCollection>("AlgoB");
    produces<TPtoRecoTrackCollection>("TP");
}


TrackAlgoCompareUtil::~TrackAlgoCompareUtil()
{
}


// ------------ method called to produce the data  ------------
void
TrackAlgoCompareUtil::produce(edm::StreamID, edm::Event& iEvent, const edm::EventSetup& iSetup) const
{
     // create output collection instance
    auto outputAlgoA = std::make_unique<RecoTracktoTPCollection>();
    auto outputAlgoB = std::make_unique<RecoTracktoTPCollection>();
    auto outputTP = std::make_unique<TPtoRecoTrackCollection>();
  
    // Get Inputs
    edm::Handle<reco::BeamSpot> recoBeamSpotHandle;
    iEvent.getByToken(beamSpotLabel, recoBeamSpotHandle);
    reco::BeamSpot beamSpot = *recoBeamSpotHandle; 
  
    edm::Handle<View<reco::Track> > trackCollAlgoA;
    iEvent.getByToken(trackLabel_algoA, trackCollAlgoA);
  
    edm::Handle< View<reco::Track> > trackCollAlgoB;
    iEvent.getByToken(trackLabel_algoB, trackCollAlgoB);
  
    edm::Handle<TrackingParticleCollection> trackingParticleCollFakes;
    iEvent.getByToken(trackingParticleLabel_fakes, trackingParticleCollFakes);
  
    edm::Handle<TrackingParticleCollection> trackingParticleCollEffic;
    iEvent.getByToken(trackingParticleLabel_effic, trackingParticleCollEffic);
  
    edm::Handle<reco::VertexCollection> vertexCollAlgoA;
    edm::Handle<reco::VertexCollection> vertexCollAlgoB;
    if(UseVertex) 
    {
        iEvent.getByToken(vertexLabel_algoA, vertexCollAlgoA);
        iEvent.getByToken(vertexLabel_algoB, vertexCollAlgoB);
    }
  
    // call the associator functions:
    reco::RecoToSimCollection recSimColl_AlgoA; 
    reco::RecoToSimCollection recSimColl_AlgoB;
  
    reco::SimToRecoCollection simRecColl_AlgoA;
    reco::SimToRecoCollection simRecColl_AlgoB;

    if(UseAssociators)
    {
        edm::Handle<reco::TrackToTrackingParticleAssociator> theAssociator_algoA;
        iEvent.getByToken(assocLabel_algoA, theAssociator_algoA);
  
        edm::Handle<reco::TrackToTrackingParticleAssociator> theAssociator_algoB;
        iEvent.getByToken(assocLabel_algoB, theAssociator_algoB);
  
        recSimColl_AlgoA = theAssociator_algoA->associateRecoToSim(trackCollAlgoA, trackingParticleCollFakes);
        recSimColl_AlgoB = theAssociator_algoB->associateRecoToSim(trackCollAlgoB, trackingParticleCollFakes);

        simRecColl_AlgoA = theAssociator_algoA->associateSimToReco(trackCollAlgoA, trackingParticleCollEffic);
        simRecColl_AlgoB = theAssociator_algoB->associateSimToReco(trackCollAlgoB, trackingParticleCollEffic);
    }
    else
    {
        Handle<reco::RecoToSimCollection > recotosimCollectionH_AlgoA;
        iEvent.getByToken(associatormap_algoA_recoToSim,recotosimCollectionH_AlgoA);
        recSimColl_AlgoA  = *(recotosimCollectionH_AlgoA.product());
        
        Handle<reco::RecoToSimCollection > recotosimCollectionH_AlgoB;
        iEvent.getByToken(associatormap_algoB_recoToSim,recotosimCollectionH_AlgoB);
        recSimColl_AlgoB  = *(recotosimCollectionH_AlgoB.product());
        
        Handle<reco::SimToRecoCollection > simtorecoCollectionH_AlgoA;
        iEvent.getByToken(associatormap_algoA_simToReco, simtorecoCollectionH_AlgoA);
        simRecColl_AlgoA = *(simtorecoCollectionH_AlgoA.product());

        Handle<reco::SimToRecoCollection > simtorecoCollectionH_AlgoB;
        iEvent.getByToken(associatormap_algoB_simToReco, simtorecoCollectionH_AlgoB);
        simRecColl_AlgoB = *(simtorecoCollectionH_AlgoB.product());
    }
    
    // define the vector of references to trackingParticleColl associated with a given reco::Track
    std::vector<std::pair<TrackingParticleRef, double> > associatedTrackingParticles;
  
    // define the vector of references to trackColl associated with a given TrackingParticle
    std::vector<std::pair<reco::TrackBaseRef, double> > associatedRecoTracks;
  
    // Get the magnetic field data from the event (used to calculate the point of closest TrackingParticle)
    edm::ESHandle<MagneticField> theMagneticField;
    iSetup.get<IdealMagneticFieldRecord>().get(theMagneticField);
    const MagneticField *magneticField = theMagneticField.product();
    
    // fill collection algoA
    for(View<reco::Track>::size_type i = 0; i < trackCollAlgoA->size(); ++i)
    {
        // get recoTrack algo A
        reco::TrackBaseRef recoTrack(trackCollAlgoA, i);
        RecoTracktoTP  recoTracktoTP;
        recoTracktoTP.SetRecoTrack(recoTrack);
        recoTracktoTP.SetBeamSpot(beamSpot.position());
        
        // get the associated trackingParticle
        if(recSimColl_AlgoA.find(recoTrack) != recSimColl_AlgoA.end())
        {
            associatedTrackingParticles = recSimColl_AlgoA[recoTrack];
            recoTracktoTP.SetTrackingParticle( associatedTrackingParticles.begin()->first );
            recoTracktoTP.SetShared( associatedTrackingParticles.begin()->second );
            SetTrackingParticleD0Dz(associatedTrackingParticles.begin()->first, beamSpot, magneticField, recoTracktoTP);
        }
        else
        {           
            recoTracktoTP.SetTrackingParticle(TrackingParticleRef());
            recoTracktoTP.SetShared(-1.0);
        }       
        
        // get the reco primary vertex info
        if(UseVertex && !vertexCollAlgoA->empty())  
        {
            recoTracktoTP.SetRecoVertex( reco::VertexRef(vertexCollAlgoA, 0) );
        }
        else
        {
            recoTracktoTP.SetRecoVertex( reco::VertexRef() );
        }   
       
        outputAlgoA->push_back(recoTracktoTP);
    }
  
  
    // fill collection algoB
    for(reco::TrackCollection::size_type i = 0; i < trackCollAlgoB->size(); ++i)
    {
        // get recoTrack algo B
        reco::TrackBaseRef recoTrack(trackCollAlgoB, i);
        RecoTracktoTP  recoTracktoTP;
        recoTracktoTP.SetRecoTrack(recoTrack);
        recoTracktoTP.SetBeamSpot(beamSpot.position());
      
        // get the associated trackingParticle
        if(recSimColl_AlgoB.find(recoTrack) != recSimColl_AlgoB.end())
        {
            associatedTrackingParticles = recSimColl_AlgoB[recoTrack];
            recoTracktoTP.SetTrackingParticle( associatedTrackingParticles.begin()->first );
            recoTracktoTP.SetShared( associatedTrackingParticles.begin()->second );
            SetTrackingParticleD0Dz(associatedTrackingParticles.begin()->first, beamSpot, magneticField, recoTracktoTP);
        }
        else
        {           
            recoTracktoTP.SetTrackingParticle(TrackingParticleRef());
            recoTracktoTP.SetShared(-1.0);
        }       
        
        // get the reco primary vertex info
        if(UseVertex && !vertexCollAlgoB->empty())  
        {
            recoTracktoTP.SetRecoVertex( reco::VertexRef(vertexCollAlgoB, 0) );
        }
        else
        {
            recoTracktoTP.SetRecoVertex( reco::VertexRef() );
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
      
        // get the assocated recoTrack algoA
        if(simRecColl_AlgoA.find(tparticle) != simRecColl_AlgoA.end())
        {
            associatedRecoTracks = simRecColl_AlgoA[tparticle];
            tptoRecoTrack.SetRecoTrack_AlgoA(associatedRecoTracks.begin()->first );
            tptoRecoTrack.SetShared_AlgoA(associatedRecoTracks.begin()->second );
        }
        else
        {
            tptoRecoTrack.SetRecoTrack_AlgoA(reco::TrackBaseRef());
            tptoRecoTrack.SetShared_AlgoA(-1.0);
        }
        
        // get the recoVertex algo A
        if(UseVertex && !vertexCollAlgoA->empty())
        {
            tptoRecoTrack.SetRecoVertex_AlgoA( reco::VertexRef(vertexCollAlgoA, 0) );
        }
        else
        {
            tptoRecoTrack.SetRecoVertex_AlgoA( reco::VertexRef() );
        }
      
        // get the assocated recoTrack algoB
        if(simRecColl_AlgoB.find(tparticle) != simRecColl_AlgoB.end())
        {
            associatedRecoTracks = simRecColl_AlgoB[tparticle];
            tptoRecoTrack.SetRecoTrack_AlgoB(associatedRecoTracks.begin()->first );
            tptoRecoTrack.SetShared_AlgoB(associatedRecoTracks.begin()->second );
        }
        else
        {
            tptoRecoTrack.SetRecoTrack_AlgoB(reco::TrackBaseRef());
            tptoRecoTrack.SetShared_AlgoB(-1.0);
        }
        // get the recoVertex algo B
        if(UseVertex && !vertexCollAlgoB->empty())
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
    iEvent.put(std::move(outputAlgoA), "AlgoA");
    iEvent.put(std::move(outputAlgoB), "AlgoB");
    iEvent.put(std::move(outputTP), "TP");
}

// ------------ Producer Specific Meber Fucntions ----------------------------------------
void TrackAlgoCompareUtil::SetTrackingParticleD0Dz(TrackingParticleRef tp, const reco::BeamSpot &bs, const MagneticField *bf, TPtoRecoTrack& TPRT) const
{
    GlobalPoint trackingParticleVertex( tp->vertex().x(), tp->vertex().y(), tp->vertex().z() );
    GlobalVector trackingParticleP3(tp->g4Track_begin()->momentum().x(),
                                    tp->g4Track_begin()->momentum().y(),
                                    tp->g4Track_begin()->momentum().z() );
    TrackCharge trackingParticleCharge(tp->charge());
  
    FreeTrajectoryState ftsAtProduction( trackingParticleVertex, trackingParticleP3, trackingParticleCharge, bf );
    TSCBLBuilderNoMaterial tscblBuilder;
    TrajectoryStateClosestToBeamLine tsAtClosestApproach = tscblBuilder(ftsAtProduction, bs);  //as in TrackProducerAlgorithm
  
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


void TrackAlgoCompareUtil::SetTrackingParticleD0Dz(TrackingParticleRef tp, const reco::BeamSpot &bs, const MagneticField *bf, RecoTracktoTP& RTTP) const
{
    GlobalPoint trackingParticleVertex( tp->vertex().x(), tp->vertex().y(), tp->vertex().z() );
    GlobalVector trackingParticleP3(tp->g4Track_begin()->momentum().x(),
                                    tp->g4Track_begin()->momentum().y(),
                                    tp->g4Track_begin()->momentum().z() );
    TrackCharge trackingParticleCharge(tp->charge());

    FreeTrajectoryState ftsAtProduction( trackingParticleVertex, trackingParticleP3, trackingParticleCharge, bf );
    TSCBLBuilderNoMaterial tscblBuilder;
    TrajectoryStateClosestToBeamLine tsAtClosestApproach = tscblBuilder(ftsAtProduction, bs);  //as in TrackProducerAlgorithm

    if(tsAtClosestApproach.isValid())
    {
        GlobalPoint v1 = tsAtClosestApproach.trackStateAtPCA().position();
        GlobalVector p = tsAtClosestApproach.trackStateAtPCA().momentum();

        RTTP.SetTrackingParticleMomentumPCA(p);
        RTTP.SetTrackingParticlePCA(v1);
    }
    else
    {
        RTTP.SetTrackingParticleMomentumPCA(GlobalVector(-9999.0, -9999.0, -9999.0));
        RTTP.SetTrackingParticlePCA(GlobalPoint(-9999.0, -9999.0, -9999.0));
    }
}

