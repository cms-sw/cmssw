#include "RecoParticleFlow/PFProducer/interface/PFProducer.h"
#include "RecoParticleFlow/PFAlgo/interface/PFGeometry.h"

// #include "DataFormats/ParticleFlowReco/interface/PFRecHit.h"
#include "DataFormats/ParticleFlowReco/interface/PFLayer.h"
#include "DataFormats/ParticleFlowReco/interface/PFCluster.h"
#include "DataFormats/ParticleFlowReco/interface/PFRecTrack.h"
#include "DataFormats/ParticleFlowReco/interface/PFRecTrackFwd.h"
#include "DataFormats/ParticleFlowReco/interface/PFParticle.h"
#include "DataFormats/ParticleFlowReco/interface/PFParticleFwd.h"

// include files used for reconstructed tracks
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrackingRecHit/interface/TrackingRecHit.h"
#include "DataFormats/TrackingRecHit/interface/TrackingRecHitFwd.h"
#include "DataFormats/TrackCandidate/interface/TrackCandidate.h"
#include "DataFormats/TrackCandidate/interface/TrackCandidateCollection.h"
#include "DataFormats/TrajectorySeed/interface/PropagationDirection.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/CommonDetUnit/interface/GeomDet.h"
#include "Geometry/Vector/interface/GlobalPoint.h"
#include "Geometry/Vector/interface/GlobalVector.h"
#include "Geometry/Surface/interface/BoundCylinder.h"
#include "Geometry/Surface/interface/TkRotation.h"
#include "Geometry/Surface/interface/SimpleCylinderBounds.h"
#include "TrackingTools/Records/interface/TrackingComponentsRecord.h" 
#include "TrackingTools/Records/interface/TransientRecHitRecord.h"  
#include "TrackingTools/PatternTools/interface/Trajectory.h"
#include "TrackingTools/PatternTools/interface/TrajectoryFitter.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"
#include "TrackingTools/GeomPropagators/interface/Propagator.h"
#include "TrackingTools/GeomPropagators/interface/AnalyticalPropagator.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "FastSimulation/Event/interface/FSimEvent.h"

using namespace std;
using namespace edm;

PFProducer::PFProducer(const edm::ParameterSet& iConfig) :
  trackAlgo_(iConfig)
{
  edm::LogInfo("PFProducer") << "Constructor" << std::endl;

  // use configuration file to setup input/output collection names
  tcCollection_ 
    = iConfig.getUntrackedParameter<std::string>("TrackCandidateCollection","TrackCandidateCollection");
  pfRecTrackCollection_ 
    = iConfig.getUntrackedParameter<std::string>("PFRecTrackCollection","PFRecTrackCollection");
  pfParticleCollection_ 
    = iConfig.getUntrackedParameter<std::string>("PFParticleCollection","PFParticleCollection");

  // register your products
  produces<reco::PFParticleCollection>(pfParticleCollection_);
  produces<reco::PFRecTrackCollection>(pfRecTrackCollection_);

  // set algorithms used for track reconstruction
  fitterName_ = iConfig.getParameter<std::string>("Fitter");   
  propagatorName_ = iConfig.getParameter<std::string>("Propagator");
  builderName_ = iConfig.getParameter<std::string>("TTRHBuilder");   

  vertexGenerator_ = iConfig.getParameter<edm::ParameterSet>
    ( "VertexGenerator" );   
  particleFilter_ = iConfig.getParameter<edm::ParameterSet>
    ( "ParticleFilter" );   

  // initialize geometry parameters
  PFGeometry pfGeometry;

  // register your products
  produces<reco::PFRecTrackCollection>(pfRecTrackCollection_);

  // dummy... just to be able to run
  // produces<reco::PFRecHitCollection >();  
}


PFProducer::~PFProducer() {
  // do anything here that needs to be done at desctruction time
  // (e.g. close files, deallocate resources etc.) 
}


void PFProducer::produce(edm::Event& iEvent, 
			 const edm::EventSetup& iSetup) 
{
  edm::LogInfo("PFProducer") << "Produce event: " << iEvent.id().event()
			     <<" in run " << iEvent.id().run() << std::endl;

  //
  // Create empty output collections
  //
  std::auto_ptr< reco::PFRecTrackCollection > pOutputPFRecTrackCollection(new reco::PFRecTrackCollection);
   
  std::auto_ptr< reco::PFParticleCollection > pOutputPFParticleCollection(new reco::PFParticleCollection ); 

  //
  // Declare and get stuff to be retrieved from ES
  //
  LogDebug("PFProducer") << "get tracker geometry" << "\n";
  edm::ESHandle<TrackerGeometry> theG;
  iSetup.get<TrackerDigiGeometryRecord>().get(theG);

  LogDebug("PFProducer") << "get magnetic field" << "\n";
  edm::ESHandle<MagneticField> theMF;
  iSetup.get<IdealMagneticFieldRecord>().get(theMF);

  LogDebug("PFProducer") << "get the trajectory fitter from the ES" << "\n";
  edm::ESHandle<TrajectoryFitter> theFitter;
  iSetup.get<TrackingComponentsRecord>().get(fitterName_, theFitter);

  LogDebug("PFProducer") << "get the trajectory propagator from the ES" << "\n";
  edm::ESHandle<Propagator> thePropagator;
  iSetup.get<TrackingComponentsRecord>().get(propagatorName_, thePropagator);

  LogDebug("PFProducer") << "get the TransientTrackingRecHitBuilder" << "\n";
  edm::ESHandle<TransientTrackingRecHitBuilder> theBuilder;
  iSetup.get<TransientRecHitRecord>().get(builderName_, theBuilder);

  //
  // Prepare propagation tools and layers
  //
  const MagneticField * magField = theMF.product();
  AnalyticalPropagator fwdPropagator(magField, alongMomentum);
  AnalyticalPropagator bkwdPropagator(magField, oppositeToMomentum);
  ReferenceCountingPointer<Surface> beamPipe(new BoundCylinder(GlobalPoint(0.,0.,0.), TkRotation<float>(), SimpleCylinderBounds(PFGeometry::innerRadius(PFGeometry::BeamPipe), PFGeometry::innerRadius(PFGeometry::BeamPipe), -1.*PFGeometry::outerZ(PFGeometry::BeamPipe), PFGeometry::outerZ(PFGeometry::BeamPipe))));
  ReferenceCountingPointer<Surface> ecalInnerWall(new BoundCylinder(GlobalPoint(0.,0.,0.), TkRotation<float>(), SimpleCylinderBounds(PFGeometry::innerRadius(PFGeometry::ECALBarrel), PFGeometry::innerRadius(PFGeometry::ECALBarrel), -1.*PFGeometry::innerZ(PFGeometry::ECALEndcap), PFGeometry::innerZ(PFGeometry::ECALEndcap))));
  ReferenceCountingPointer<Surface> ps1Wall(new BoundCylinder(GlobalPoint(0.,0.,0.), TkRotation<float>(), SimpleCylinderBounds(PFGeometry::innerRadius(PFGeometry::ECALBarrel), PFGeometry::innerRadius(PFGeometry::ECALBarrel), -1.*PFGeometry::innerZ(PFGeometry::PS1), PFGeometry::innerZ(PFGeometry::PS1))));
  ReferenceCountingPointer<Surface> ps2Wall(new BoundCylinder(GlobalPoint(0.,0.,0.), TkRotation<float>(), SimpleCylinderBounds(PFGeometry::innerRadius(PFGeometry::ECALBarrel), PFGeometry::innerRadius(PFGeometry::ECALBarrel), -1.*PFGeometry::innerZ(PFGeometry::PS2), PFGeometry::innerZ(PFGeometry::PS2))));
  ReferenceCountingPointer<Surface> hcalInnerWall(new BoundCylinder(GlobalPoint(0.,0.,0.), TkRotation<float>(), SimpleCylinderBounds(PFGeometry::innerRadius(PFGeometry::HCALBarrel), PFGeometry::innerRadius(PFGeometry::HCALBarrel), -1.*PFGeometry::innerZ(PFGeometry::HCALEndcap), PFGeometry::innerZ(PFGeometry::HCALEndcap))));
  ReferenceCountingPointer<Surface> hcalOuterWall(new BoundCylinder(GlobalPoint(0.,0.,0.), TkRotation<float>(), SimpleCylinderBounds(PFGeometry::outerRadius(PFGeometry::HCALBarrel), PFGeometry::outerRadius(PFGeometry::HCALBarrel), -1.*PFGeometry::outerZ(PFGeometry::HCALEndcap), PFGeometry::outerZ(PFGeometry::HCALEndcap))));

  //
  // Get track candidates and create smoothed tracks
  // Temporary solution. Let's hope that in the future, Trajectory objects 
  // will become persistent.
  //
  AlgoProductCollection algoResults;
  try{
    LogDebug("PFProducer") << "get the TrackCandidateCollection"
			   << " from the event, source is " 
			   << tcCollection_ <<"\n";
    edm::Handle<TrackCandidateCollection> theTCCollection;
    iEvent.getByLabel(tcCollection_, theTCCollection);

    //
    //run the algorithm  
    //
    LogDebug("PFProducer") << "run the tracking algorithm" << "\n";
    trackAlgo_.runWithCandidate(theG.product(), theMF.product(), 
				*theTCCollection,
				theFitter.product(), thePropagator.product(), 
				theBuilder.product(), algoResults);
  } catch (cms::Exception& e) { 
    edm::LogError("PFProducer") << "cms::Exception caught : " 
				<< "cannot get collection " 
				<< tcCollection_ << "\n" << e << "\n";
  }

  //
  // Loop over smoothed tracks and fill PFRecTrack collection
  //
  for(AlgoProductCollection::iterator itTrack = algoResults.begin();
      itTrack != algoResults.end(); itTrack++) {
    Trajectory*  theTraj  = (*itTrack).first;
    std::vector<TrajectoryMeasurement> measurements = theTraj->measurements();
    reco::Track* theTrack = (*itTrack).second;
    reco::PFRecTrack track(theTrack->parameters().charge(), 
			   reco::PFRecTrack::KF);

    // Closest approach of the beamline
    math::XYZPoint posClosest(theTrack->x(), theTrack->y(), theTrack->z());
    math::XYZTLorentzVector momClosest(theTrack->px(), theTrack->py(), 
				       theTrack->pz(), theTrack->p());
    reco::PFTrajectoryPoint closestPt(0, 
				      reco::PFTrajectoryPoint::ClosestApproach,
				      posClosest, momClosest);
    track.addPoint(closestPt);
    LogDebug("PFProducer") << "closest approach point " << closestPt << "\n";

    if (posClosest.Rho() < PFGeometry::innerRadius(PFGeometry::BeamPipe)) {
      // Intersection with beam pipe
      TrajectoryStateOnSurface innerTSOS;
      if (theTraj->direction() == alongMomentum)
	innerTSOS = measurements[0].updatedState();
      else
	innerTSOS = measurements[measurements.size() - 1].updatedState();
      TrajectoryStateOnSurface beamPipeTSOS = 
	bkwdPropagator.propagate(innerTSOS, *beamPipe);
      GlobalPoint vBeamPipe  = beamPipeTSOS.globalParameters().position();
      GlobalVector pBeamPipe = beamPipeTSOS.globalParameters().momentum();
      math::XYZPoint posBeamPipe(vBeamPipe.x(), vBeamPipe.y(), vBeamPipe.z());
      math::XYZTLorentzVector momBeamPipe(pBeamPipe.x(), pBeamPipe.y(), 
					  pBeamPipe.z(), pBeamPipe.mag());
      reco::PFTrajectoryPoint beamPipePt(0, reco::PFTrajectoryPoint::BeamPipe, 
					 posBeamPipe, momBeamPipe);
      track.addPoint(beamPipePt);
      LogDebug("PFProducer") << "beam pipe point " << beamPipePt << "\n";
    }

    //
    // Loop over trajectory measurements
    //
    // Order measurements along momentum
    int iTrajFirst = 0;
    int iTrajLast  = measurements.size();
    int increment = +1;
    if (theTraj->direction() == oppositeToMomentum) {
      iTrajFirst = measurements.size() - 1;
      iTrajLast = -1;
      increment = -1;
    }
    for (int iTraj = iTrajFirst; iTraj != iTrajLast; iTraj += increment) {
      TrajectoryStateOnSurface tsos = measurements[iTraj].updatedState();
      GlobalPoint v  = tsos.globalParameters().position();
      GlobalVector p = tsos.globalParameters().momentum();
      math::XYZPoint  pos(v.x(), v.y(), v.z());       
      math::XYZTLorentzVector mom(p.x(), p.y(), p.z(), p.mag());
      unsigned int detId = 
	measurements[iTraj].recHit()->det()->geographicalId().rawId();
      reco::PFTrajectoryPoint trajPt(detId, reco::PFTrajectoryPoint::NLayers, 
				     pos, mom);
      track.addPoint(trajPt);
      LogDebug("PFProducer") << "add measuremnt " << iTraj << " " << trajPt << "\n";
    }

    // Propagate track to ECAL entrance
    TrajectoryStateOnSurface outerTSOS;
    if (theTraj->direction() == alongMomentum)
      outerTSOS = measurements[measurements.size() - 1].updatedState();
    else
      outerTSOS = measurements[0].updatedState();
    TrajectoryStateOnSurface ecalTSOS = 
      fwdPropagator.propagate(outerTSOS, *ecalInnerWall);
    GlobalPoint vECAL  = ecalTSOS.globalParameters().position();
    GlobalVector pECAL = ecalTSOS.globalParameters().momentum();
    math::XYZPoint posECAL(vECAL.x(), vECAL.y(), vECAL.z());       
    math::XYZTLorentzVector momECAL(pECAL.x(), pECAL.y(), pECAL.z(), 
				    pECAL.mag());
    reco::PFTrajectoryPoint ecalPt(0, reco::PFTrajectoryPoint::ECALEntrance, 
				   posECAL, momECAL);
    bool isBelowPS = false;
    if (posECAL.Rho() < PFGeometry::innerRadius(PFGeometry::ECALBarrel)) {
      // Propagate track to preshower layer1
      TrajectoryStateOnSurface ps1TSOS = 
	fwdPropagator.propagate(outerTSOS, *ps1Wall);
      GlobalPoint vPS1  = ps1TSOS.globalParameters().position();
      GlobalVector pPS1 = ps1TSOS.globalParameters().momentum();
      math::XYZPoint posPS1(vPS1.x(), vPS1.y(), vPS1.z());
      if (posPS1.Rho() >= PFGeometry::innerRadius(PFGeometry::PS1) &&
	  posPS1.Rho() <= PFGeometry::outerRadius(PFGeometry::PS1)) {
	isBelowPS = true;
	math::XYZTLorentzVector momPS1(pPS1.x(), pPS1.y(), pPS1.z(), 
				       pPS1.mag());
	reco::PFTrajectoryPoint ps1Pt(0, reco::PFTrajectoryPoint::PS1, 
				      posPS1, momPS1);
	track.addPoint(ps1Pt);
	LogDebug("PFProducer") << "ps1 point " << ps1Pt << "\n";
      } else {
	reco::PFTrajectoryPoint dummyPS1;
	track.addPoint(dummyPS1);
      }

      // Propagate track to preshower layer2
      TrajectoryStateOnSurface ps2TSOS = 
	fwdPropagator.propagate(outerTSOS, *ps2Wall);
      GlobalPoint vPS2  = ps2TSOS.globalParameters().position();
      GlobalVector pPS2 = ps2TSOS.globalParameters().momentum();
      math::XYZPoint posPS2(vPS2.x(), vPS2.y(), vPS2.z());
      if (posPS2.Rho() >= PFGeometry::innerRadius(PFGeometry::PS2) &&
	  posPS2.Rho() <= PFGeometry::outerRadius(PFGeometry::PS2)) {
	isBelowPS = true;
	math::XYZTLorentzVector momPS2(pPS2.x(), pPS2.y(), pPS2.z(), 
				       pPS2.mag());
	reco::PFTrajectoryPoint ps2Pt(0, reco::PFTrajectoryPoint::PS2, 
				      posPS2, momPS2);
	track.addPoint(ps2Pt);
	LogDebug("PFProducer") << "ps2 point " << ps2Pt << "\n";
      } else {
	reco::PFTrajectoryPoint dummyPS2;
	track.addPoint(dummyPS2);
      }
    } else {
      // add dummy point for PS1 and PS2
      reco::PFTrajectoryPoint dummyPS1;
      reco::PFTrajectoryPoint dummyPS2;
      track.addPoint(dummyPS1);
      track.addPoint(dummyPS2);
    }
    track.addPoint(ecalPt);
    LogDebug("PFProducer") << "ecal point " << ecalPt << "\n";

    // Propage track to ECAL shower max TODO
    // Be careful : the following formula are only valid for electrons !
    double ecalShowerDepth = reco::PFCluster::getDepthCorrection(momECAL.E(), 
								 isBelowPS, 
								 false);
    math::XYZPoint showerDirection(momECAL.Px(), momECAL.Py(), momECAL.Pz());
    showerDirection *= ecalShowerDepth/showerDirection.R();
    double rCyl = PFGeometry::innerRadius(PFGeometry::ECALBarrel) + 
      showerDirection.Rho();
    double zCyl = PFGeometry::innerZ(PFGeometry::ECALEndcap) + 
      fabs(showerDirection.Z());
    ReferenceCountingPointer<Surface> showerMaxWall(new BoundCylinder(GlobalPoint(0.,0.,0.), TkRotation<float>(), SimpleCylinderBounds(rCyl, rCyl, -1.*zCyl, zCyl)));
    TrajectoryStateOnSurface showerMaxTSOS = 
      fwdPropagator.propagate(ecalTSOS, *showerMaxWall);
    GlobalPoint vShowerMax  = showerMaxTSOS.globalParameters().position();
    GlobalVector pShowerMax = showerMaxTSOS.globalParameters().momentum();
    math::XYZPoint posShowerMax(vShowerMax.x(), vShowerMax.y(), 
				vShowerMax.z());
    math::XYZTLorentzVector momShowerMax(pShowerMax.x(), pShowerMax.y(), 
					 pShowerMax.z(), pShowerMax.mag());
    reco::PFTrajectoryPoint ecalShowerMaxPt(0, reco::PFTrajectoryPoint::ECALShowerMax, 
					    posShowerMax, momShowerMax);
    track.addPoint(ecalShowerMaxPt);
    LogDebug("PFProducer") << "ecal shower maximum point " << ecalShowerMaxPt 
			   << "\n";    
    
    // Propagate track to HCAL entrance
    TrajectoryStateOnSurface hcalTSOS = 
      fwdPropagator.propagate(ecalTSOS, *hcalInnerWall);
    GlobalPoint vHCAL  = hcalTSOS.globalParameters().position();
    GlobalVector pHCAL = hcalTSOS.globalParameters().momentum();
    math::XYZPoint posHCAL(vHCAL.x(), vHCAL.y(), vHCAL.z());       
    math::XYZTLorentzVector momHCAL(pHCAL.x(), pHCAL.y(), pHCAL.z(), 
				    pHCAL.mag());
    reco::PFTrajectoryPoint hcalPt(0, reco::PFTrajectoryPoint::HCALEntrance, 
				   posHCAL, momHCAL);
    track.addPoint(hcalPt);
    LogDebug("PFProducer") << "hcal point " << hcalPt << "\n";    

    // Propagate track to HCAL exit
    TrajectoryStateOnSurface hcalExitTSOS = 
      fwdPropagator.propagate(hcalTSOS, *hcalOuterWall);
    GlobalPoint vHCALExit  = hcalExitTSOS.globalParameters().position();
    GlobalVector pHCALExit = hcalExitTSOS.globalParameters().momentum();
    math::XYZPoint posHCALExit(vHCALExit.x(), vHCALExit.y(), vHCALExit.z());
    math::XYZTLorentzVector momHCALExit(pHCALExit.x(), pHCALExit.y(), 
					pHCALExit.z(), pHCALExit.mag());
    reco::PFTrajectoryPoint hcalExitPt(0, reco::PFTrajectoryPoint::HCALExit, 
				       posHCALExit, momHCALExit);
    track.addPoint(hcalExitPt);
    LogDebug("PFProducer") << "hcal exit point " << hcalExitPt << "\n";    
    
    pOutputPFRecTrackCollection->push_back(track);
   
    LogDebug("PFProducer") << "Add a new PFRecTrack " << track << "\n";
  }

  // deal with particles 
  FSimEvent mySimEvent = 
    FSimEvent( vertexGenerator_, particleFilter_);
  
  edm::Handle<std::vector<SimTrack> > simtracks;
  iEvent.getByLabel("SimG4Object",simtracks);
  edm::Handle<std::vector<SimVertex> > simvertices;
  iEvent.getByLabel("SimG4Object",simvertices);

  for(unsigned it = 0; it<simtracks->size(); it++ ) {
    cout<<"\t track "<< (*simtracks)[it]<<endl;
  }

//   mySimEvent.fill( *simtracks, *simvertices );



  pOutputPFParticleCollection->push_back( reco::PFParticle(-1, 11, 1, 2, 3) );

  //
  // Put the products in the event
  //
//   edm::LogInfo("PFProducer") << " Put the PFRecTrackCollection of " 
// 			     << pOutputPFRecTrackCollection->size() 
// 			     << " candidates in the Event" << std::endl;
//   pOutputPFRecTrackCollection->assign(outputPFRecTrackCollection.begin(),
//  				      outputPFRecTrackCollection.end());
  iEvent.put(pOutputPFRecTrackCollection, pfRecTrackCollection_);
  iEvent.put(pOutputPFParticleCollection, pfParticleCollection_);

}


//define this as a plug-in
DEFINE_FWK_MODULE(PFProducer)
