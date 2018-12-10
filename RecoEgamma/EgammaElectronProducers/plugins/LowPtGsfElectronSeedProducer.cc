// -*- C++ -*-
//
// Package:    ElectronProducers
// Class:      LowPtGsfElectronSeedProducer
//
/**\class LowPtGsfElectronSeedProducer RecoEgamma/EgammaElectronProducers/plugins/LowPtGsfElectronSeedProducer.cc
 Description: EDProducer of ElectronSeed objects
 Implementation:
     <Notes on implementation>
*/
// Original Author:  Robert Bainbridge

#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "DataFormats/GeometryVector/interface/GlobalVector.h"
#include "DataFormats/Math/interface/deltaPhi.h"
#include "DataFormats/Math/interface/deltaR.h"
#include "DataFormats/Math/interface/LorentzVector.h"
#include "DataFormats/ParticleFlowReco/interface/PFTrajectoryPoint.h"
#include "DataFormats/TrackReco/interface/TrackBase.h"
#include "FastSimulation/BaseParticlePropagator/interface/BaseParticlePropagator.h"
#include "FastSimulation/Particle/interface/RawParticle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "RecoEgamma/EgammaElectronProducers/plugins/LowPtGsfElectronSeedProducer.h"
#include "RecoTracker/TransientTrackingRecHit/interface/TkClonerImpl.h"
#include "RecoTracker/TransientTrackingRecHit/interface/TkTransientTrackingRecHitBuilder.h"
#include "TrackingTools/PatternTools/interface/Trajectory.h"
#include "TrackingTools/PatternTools/interface/TrajectorySmoother.h"
#include "TrackingTools/Records/interface/TransientRecHitRecord.h"
#include "TrackingTools/TrackFitters/interface/TrajectoryFitter.h"
#include "TrackingTools/TrajectoryParametrization/interface/GlobalTrajectoryParameters.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"
#include "TMath.h"

//////////////////////////////////////////////////////////////////////////////////////////
//
LowPtGsfElectronSeedProducer::LowPtGsfElectronSeedProducer( const edm::ParameterSet& conf, 
							    const lowptgsfeleseed::HeavyObjectCache* ) :
  ecalClusters_{consumes<reco::PFClusterCollection>(conf.getParameter<edm::InputTag>("ecalClusters"))},
  hcalClusters_{consumes<reco::PFClusterCollection>(conf.getParameter<edm::InputTag>("hcalClusters"))},
  ebRecHits_{consumes<EcalRecHitCollection>(conf.getParameter<edm::InputTag>("EBRecHits"))},
  eeRecHits_{consumes<EcalRecHitCollection>(conf.getParameter<edm::InputTag>("EERecHits"))},
  rho_(consumes<double>(conf.getParameter<edm::InputTag>("rho"))),
  beamSpot_(consumes<reco::BeamSpot>(conf.getParameter<edm::InputTag>("BeamSpot"))),
  fitter_(conf.getParameter<std::string>("Fitter")),
  smoother_(conf.getParameter<std::string>("Smoother")),
  builder_(conf.getParameter<std::string>("TTRHBuilder")),
  passThrough_(conf.getParameter<bool>("PassThrough")),
  usePfTracks_(conf.getParameter<bool>("UsePfTracks")),
  minPtThreshold_(conf.getParameter<double>("MinPtThreshold")),
  maxPtThreshold_(conf.getParameter<double>("MaxPtThreshold"))
{
  if ( usePfTracks_ ) { pfTracks_ = consumes<reco::PFRecTrackCollection>(conf.getParameter<edm::InputTag>("pfTracks")); }
  else                { kfTracks_ = consumes<reco::TrackCollection>(conf.getParameter<edm::InputTag>("tracks")); }
  produces<reco::ElectronSeedCollection>();
  produces<reco::PreIdCollection>();
  produces<reco::PreIdCollection>("HCAL");
}

//////////////////////////////////////////////////////////////////////////////////////////
//
LowPtGsfElectronSeedProducer::~LowPtGsfElectronSeedProducer() {}

//////////////////////////////////////////////////////////////////////////////////////////
//
void LowPtGsfElectronSeedProducer::beginLuminosityBlock( edm::LuminosityBlock const&, 
							 edm::EventSetup const& setup ) 
{
  setup.get<IdealMagneticFieldRecord>().get(field_);
}

//////////////////////////////////////////////////////////////////////////////////////////
//
void LowPtGsfElectronSeedProducer::produce( edm::Event& event, 
					    const edm::EventSetup& setup ) 
{
  
  // Products
  auto seeds = std::make_unique<reco::ElectronSeedCollection>();
  auto ecalPreIds = std::make_unique<reco::PreIdCollection>();
  auto hcalPreIds = std::make_unique<reco::PreIdCollection>();
  
  // KF tracks
  edm::Handle<reco::TrackCollection> kfTracks;
  if ( !usePfTracks_ ) { event.getByToken(kfTracks_, kfTracks); }

  // PF tracks
  edm::Handle<reco::PFRecTrackCollection> pfTracks;
  if ( usePfTracks_ ) { event.getByToken(pfTracks_, pfTracks); }

  // ECAL clusters
  edm::Handle<reco::PFClusterCollection> ecalClusters;
  event.getByToken(ecalClusters_,ecalClusters);

  // HCAL clusters (only used with PF tracks)
  edm::Handle<reco::PFClusterCollection> hcalClusters;
  event.getByToken(hcalClusters_,hcalClusters);

  if ( usePfTracks_ ) { 
    loop(pfTracks, // PF tracks
	 ecalClusters,
	 hcalClusters,
	 *seeds,
	 *ecalPreIds,
	 *hcalPreIds,
	 event,
	 setup);
  } else { 
    loop(kfTracks, // KF tracks
	 ecalClusters,
	 hcalClusters,
	 *seeds,
	 *ecalPreIds,
	 *hcalPreIds,
	 event,
	 setup);
  }

  event.put(std::move(seeds));
  event.put(std::move(ecalPreIds));
  event.put(std::move(hcalPreIds),"HCAL");
  
}

//////////////////////////////////////////////////////////////////////////////////////////
// Return reco::Track from edm::Ref<T>

reco::TrackRef LowPtGsfElectronSeedProducer::getBaseRef( edm::Handle< std::vector<reco::Track> > handle, int idx ) const 
{
  return reco::TrackRef(handle,idx);
}

reco::TrackRef LowPtGsfElectronSeedProducer::getBaseRef( edm::Handle< std::vector<reco::PFRecTrack> > handle, int idx ) const
{
  return reco::PFRecTrackRef(handle,idx)->trackRef();
}

//////////////////////////////////////////////////////////////////////////////////////////
// Template function, instantiated for both reco::Tracks and reco::PFRecTracks 
template <typename T>
void LowPtGsfElectronSeedProducer::loop( const edm::Handle< std::vector<T> >& handle, // PF or KF tracks
					 edm::Handle<reco::PFClusterCollection>& ecalClusters,
					 edm::Handle<reco::PFClusterCollection>& hcalClusters,
					 reco::ElectronSeedCollection& seeds,
					 reco::PreIdCollection& ecalPreIds, 
					 reco::PreIdCollection& hcalPreIds,
					 edm::Event& event,
					 const edm::EventSetup& setup )
{
  
  // Pileup
  edm::Handle<double> rho;
  event.getByToken(rho_,rho);
  
  // Beam spot
  edm::Handle<reco::BeamSpot> spot;
  event.getByToken(beamSpot_,spot);
  
  // Utility to access to shower shape vars
  noZS::EcalClusterLazyTools ecalTools(event,setup,ebRecHits_,eeRecHits_);
  
  // Ensure each cluster is only matched once to a track
  std::vector<int> matchedEcalClusters;
  std::vector<int> matchedHcalClusters;
  
  // Iterate through (PF or KF) tracks
  for ( unsigned int itrk = 0; itrk < handle.product()->size(); itrk++ ) {

    edm::Ref< std::vector<T> > templatedRef(handle,itrk); // TrackRef or PFRecTrackRef
    reco::TrackRef trackRef = getBaseRef(handle,itrk);

    if ( !(trackRef->quality(reco::TrackBase::qualityByName("highPurity"))) ) { continue; }
    if ( !passThrough_ && ( trackRef->pt() < minPtThreshold_ ) ) { continue; }

    // Create ElectronSeed 
    reco::ElectronSeed seed( *(trackRef->seedRef()) );
    seed.setCtfTrack(trackRef);
    
    // Create PreIds
    unsigned int nModels = globalCache()->modelNames().size();
    reco::PreId ecalPreId(nModels);
    reco::PreId hcalPreId(nModels);

    // Add track ref to PreId
    ecalPreId.setTrack(trackRef);
    hcalPreId.setTrack(trackRef);

    // Add Track-Calo matching variables to PreIds
    propagateTrackToCalo(templatedRef,
			 ecalClusters,
			 hcalClusters,
			 matchedEcalClusters,
			 matchedHcalClusters,
			 ecalPreId,
			 hcalPreId );
    
    // Add variables related to GSF tracks to PreId
    lightGsfTracking(ecalPreId,trackRef,seed,setup); 

    // Decision based on BDT 
    bool result = decision(templatedRef,ecalPreId,hcalPreId,*rho,*spot,ecalTools);

    // If fails BDT, do not store seed
    if ( !result ) { continue; }
    
    // Store PreId
    ecalPreIds.push_back(ecalPreId);
    hcalPreIds.push_back(hcalPreId);

    // Store ElectronSeed
    seeds.push_back(seed);

  }

}

//////////////////////////////////////////////////////////////////////////////////////////
// Template instantiation for reco::Tracks
template 
void LowPtGsfElectronSeedProducer::loop<reco::Track>( const edm::Handle< std::vector<reco::Track> >&,
						      edm::Handle<reco::PFClusterCollection>& ecalClusters,
						      edm::Handle<reco::PFClusterCollection>& hcalClusters,
						      reco::ElectronSeedCollection& seeds,
						      reco::PreIdCollection& ecalPreIds, 
						      reco::PreIdCollection& hcalPreIds,
						      edm::Event&,
						      const edm::EventSetup& );

//////////////////////////////////////////////////////////////////////////////////////////
// Template instantiation for reco::PFRecTracks
template 
void LowPtGsfElectronSeedProducer::loop<reco::PFRecTrack>( const edm::Handle< std::vector<reco::PFRecTrack> >&,
							   edm::Handle<reco::PFClusterCollection>& ecalClusters,
							   edm::Handle<reco::PFClusterCollection>& hcalClusters,
							   reco::ElectronSeedCollection& seeds,
							   reco::PreIdCollection& ecalPreIds, 
							   reco::PreIdCollection& hcalPreIds,
							   edm::Event&,
							   const edm::EventSetup& );

//////////////////////////////////////////////////////////////////////////////////////////
// Loops through both ECAL and HCAL clusters
void LowPtGsfElectronSeedProducer::propagateTrackToCalo( const reco::PFRecTrackRef& pfTrackRef,
							 const edm::Handle<reco::PFClusterCollection>& ecalClusters,
							 const edm::Handle<reco::PFClusterCollection>& hcalClusters,
							 std::vector<int>& matchedEcalClusters,
							 std::vector<int>& matchedHcalClusters,
							 reco::PreId& ecalPreId, 
							 reco::PreId& hcalPreId ) 
{
  propagateTrackToCalo( pfTrackRef, ecalClusters, matchedEcalClusters, ecalPreId, true );
  propagateTrackToCalo( pfTrackRef, hcalClusters, matchedHcalClusters, hcalPreId, false );
}

//////////////////////////////////////////////////////////////////////////////////////////
// Loops through ECAL or HCAL clusters (called twice)
void LowPtGsfElectronSeedProducer::propagateTrackToCalo( const reco::PFRecTrackRef& pfTrackRef,
							 const edm::Handle<reco::PFClusterCollection>& clusters,
							 std::vector<int>& matched,
							 reco::PreId& preId,
							 bool ecal )
{

  // Store info for PreId
  struct Info {
    reco::PFClusterRef cluRef = reco::PFClusterRef();
    float dr2min = 1.e6;
    float deta = 1.e6;
    float dphi = 1.e6;
    math::XYZPoint showerPos = math::XYZPoint(0.,0.,0.);
  } info;
  
  // Find closest "seed cluster" to KF track extrapolated to ECAL (or HCAL)
  reco::PFTrajectoryPoint point;
  if ( ecal ) { point = pfTrackRef->extrapolatedPoint(reco::PFTrajectoryPoint::LayerType::ECALShowerMax); }
  else        { point = pfTrackRef->extrapolatedPoint(reco::PFTrajectoryPoint::LayerType::HCALEntrance); }

  if ( point.isValid() ) {

    Info info;
    for ( unsigned int iclu = 0; iclu < clusters.product()->size(); iclu++ ) {

      if ( std::find( matched.begin(), matched.end(), iclu ) == matched.end() ) {
	reco::PFClusterRef cluRef(clusters,iclu);

	// Determine deta, dphi, dr
	float deta = cluRef->positionREP().eta() - point.positionREP().eta();
	float dphi = reco::deltaPhi( cluRef->positionREP().phi(), point.positionREP().phi() );
	float dr2 = reco::deltaR2( cluRef->positionREP(), point.positionREP() );

	if ( dr2 < info.dr2min ) {
	  info.dr2min = dr2;
	  info.cluRef = cluRef;
	  info.deta = deta;
	  info.dphi = dphi;
	  info.showerPos = point.position();
	}

      }
    }

    // Set PreId content if match found
    if ( info.dr2min < 1.e5 ) { 
      float ep = info.cluRef->correctedEnergy() / std::sqrt( pfTrackRef->trackRef()->innerMomentum().mag2() );
      preId.setECALMatchingProperties( info.cluRef,
				       point.position(), // ECAL or HCAL surface
				       info.showerPos, // 
				       info.deta,
				       info.dphi,
				       0.f, // chieta
				       0.f, // chiphi
				       pfTrackRef->trackRef()->normalizedChi2(), // chi2
				       ep );
    }

  } // clusters

}

//////////////////////////////////////////////////////////////////////////////////////////
// Original implementation in GoodSeedProducer, loops over ECAL clusters only
void LowPtGsfElectronSeedProducer::propagateTrackToCalo( const reco::TrackRef& kfTrackRef,
							 const edm::Handle<reco::PFClusterCollection>& ecalClusters,
							 const edm::Handle<reco::PFClusterCollection>& hcalClusters, // not used
							 std::vector<int>& matchedEcalClusters,
							 std::vector<int>& matchedHcalClusters, // not used
							 reco::PreId& ecalPreId, 
							 reco::PreId& hcalPreId /* not used */ ) 
{

  // Store info for PreId
  struct Info {
    reco::PFClusterRef cluRef = reco::PFClusterRef();
    float dr2min = 1.e6;
    float deta = 1.e6;
    float dphi = 1.e6;
    math::XYZPoint showerPos = math::XYZPoint(0.,0.,0.);
  } info;

  // Propagate 'electron' to ECAL surface
  float energy = sqrt( pow(0.000511,2.) + kfTrackRef->outerMomentum().Mag2() );
  XYZTLorentzVector mom = XYZTLorentzVector( kfTrackRef->outerMomentum().x(),
					     kfTrackRef->outerMomentum().y(),
					     kfTrackRef->outerMomentum().z(),
					     energy );
  XYZTLorentzVector pos = XYZTLorentzVector( kfTrackRef->outerPosition().x(),
					     kfTrackRef->outerPosition().y(),
					     kfTrackRef->outerPosition().z(),
					     0. );
  math::XYZVector field(field_->inTesla(GlobalPoint(0,0,0)));
  BaseParticlePropagator particle( RawParticle(mom,pos), 0, 0, field.z() );
  particle.setCharge(kfTrackRef->charge());
  particle.propagateToEcalEntrance(false);
  if ( particle.getSuccess() == 0 ) { return; }
  
  // ECAL entry point for track
  GlobalPoint ecal_pos(particle.vertex().x(),
		       particle.vertex().y(),
		       particle.vertex().z());
  // Preshower limit
  bool below_ps = pow(ecal_pos.z(),2.) > pow(2.50746495928f,2.)*ecal_pos.perp2();
  
  // Iterate through ECAL clusters 
  for ( unsigned int iclu = 0; iclu < ecalClusters.product()->size(); iclu++ ) {
    reco::PFClusterRef cluRef(ecalClusters,iclu);

    // Correct ecal_pos for shower depth 
    double shower_depth = reco::PFCluster::getDepthCorrection(cluRef->correctedEnergy(),
							      below_ps,
							      false);
    GlobalPoint showerPos = ecal_pos + 
      GlobalVector(particle.momentum().x(),
		   particle.momentum().y(),
		   particle.momentum().z()).unit() * shower_depth;

    // Determine deta, dphi, dr
    float deta = std::abs( cluRef->positionREP().eta() - showerPos.eta() );
    float dphi = std::abs( reco::deltaPhi( cluRef->positionREP().phi(), showerPos.phi() ));
    float dr2 = reco::deltaR2( cluRef->positionREP(), showerPos );

    // Find nearest ECAL cluster
    if ( dr2 < info.dr2min ) {
      info.dr2min = dr2;
      info.cluRef = cluRef;
      info.deta = deta;
      info.dphi = dphi;
      info.showerPos = showerPos;
    }
  
  }

  // Populate PreId object
  math::XYZPoint point( ecal_pos.x(),
			ecal_pos.y(),
			ecal_pos.z() );

  // Set PreId content
  ecalPreId.setECALMatchingProperties( info.cluRef,
				       point,
				       info.showerPos,
				       info.deta,
				       info.dphi,
				       0.f, // chieta
				       0.f, // chiphi
				       kfTrackRef->normalizedChi2(), // chi2
				       info.cluRef->correctedEnergy() / std::sqrt( kfTrackRef->innerMomentum().mag2() ) ); // E/p

}

//////////////////////////////////////////////////////////////////////////////////////////
// Original implementation for "lightweight" GSF tracking
bool LowPtGsfElectronSeedProducer::lightGsfTracking( reco::PreId& preId,
						     const reco::TrackRef& trackRef,
						     const reco::ElectronSeed& seed,
						     const edm::EventSetup& setup )
{
  
  edm::ESHandle<TrajectoryFitter> fitter;
  setup.get<TrajectoryFitter::Record>().get(fitter_,fitter);
  std::unique_ptr<TrajectoryFitter> fitterPtr = fitter->clone();

  edm::ESHandle<TrajectorySmoother> smoother;
  setup.get<TrajectoryFitter::Record>().get(smoother_,smoother);
  std::unique_ptr<TrajectorySmoother> smootherPtr;
  smootherPtr.reset(smoother->clone());

  edm::ESHandle<TransientTrackingRecHitBuilder> builder;
  setup.get<TransientRecHitRecord>().get(builder_,builder);
  TkClonerImpl hitCloner = static_cast<TkTransientTrackingRecHitBuilder const*>(builder.product())->cloner();
  fitterPtr->setHitCloner(&hitCloner);
  smootherPtr->setHitCloner(&hitCloner);

  Trajectory::ConstRecHitContainer hits;
  for ( unsigned int ihit = 0; ihit < trackRef->recHitsSize(); ++ihit ) {
    hits.push_back( trackRef->recHit(ihit)->cloneSH() );
  }
  
  GlobalVector gv( trackRef->innerMomentum().x(),
		   trackRef->innerMomentum().y(),
		   trackRef->innerMomentum().z() );
  GlobalPoint gp( trackRef->innerPosition().x(),
		  trackRef->innerPosition().y(),
		  trackRef->innerPosition().z() );

  GlobalTrajectoryParameters gtps( gp,
				   gv,
				   trackRef->charge(),
				   &*field_ );

  TrajectoryStateOnSurface tsos( gtps,
				 trackRef->innerStateCovariance(),
				 *hits[0]->surface() );

  // Track fitted and smoothed under electron hypothesis
  Trajectory traj1 = fitterPtr->fitOne( seed, hits, tsos );
  if ( !traj1.isValid() ) { return false; }
  Trajectory traj2 = smootherPtr->trajectory(traj1);
  if ( !traj2.isValid() ) {  return false; }

  // Set PreId content
  float chi2Ratio = traj2.chiSquared() / trackRef->chi2();
  float gsfReducedChi2 = chi2Ratio * trackRef->normalizedChi2();
  float ptOut = traj2.firstMeasurement().updatedState().globalMomentum().perp();
  float ptIn = traj2.lastMeasurement().updatedState().globalMomentum().perp();
  float gsfDpt = ( ptIn > 0 ) ? fabs( ptOut - ptIn ) / ptIn : 0.;
  preId.setTrackProperties(gsfReducedChi2,chi2Ratio,gsfDpt);

  return true;

}

//////////////////////////////////////////////////////////////////////////////////////////
// Decision based on OR of outputs from list of models
bool LowPtGsfElectronSeedProducer::decision( const reco::PFRecTrackRef& pfTrackRef,
					     reco::PreId& ecalPreId,
					     reco::PreId& hcalPreId,
					     double rho,
					     const reco::BeamSpot& spot,
					     noZS::EcalClusterLazyTools& ecalTools )
{
  bool result = false;
  for ( auto& name: globalCache()->modelNames() ) {
    result |= globalCache()->eval(name,
				  ecalPreId,
				  hcalPreId,
				  rho,
				  spot,
				  ecalTools);
  }
  return passThrough_ || ( pfTrackRef->trackRef()->pt() > maxPtThreshold_ ) || result;
}

//////////////////////////////////////////////////////////////////////////////////////////
// 
bool LowPtGsfElectronSeedProducer::decision( const reco::TrackRef& kfTrackRef,
					     reco::PreId& ecalPreId,
					     reco::PreId& hcalPreId,
					     double rho,
					     const reco::BeamSpot& spot,
					     noZS::EcalClusterLazyTools& ecalTools )
{
  // No implementation currently
  return passThrough_;
}
 
//////////////////////////////////////////////////////////////////////////////////////////
//
void LowPtGsfElectronSeedProducer::fillDescription( edm::ParameterSetDescription& desc ) 
{
  desc.add<edm::InputTag>("tracks",edm::InputTag("generalTracks"));
  desc.add<edm::InputTag>("pfTracks",edm::InputTag("lowPtGsfElePfTracks"));
  desc.add<edm::InputTag>("ecalClusters",edm::InputTag("particleFlowClusterECAL"));
  desc.add<edm::InputTag>("hcalClusters",edm::InputTag("particleFlowClusterHCAL"));
  desc.add<edm::InputTag>("EBRecHits",edm::InputTag("reducedEcalRecHitsEB"));
  desc.add<edm::InputTag>("EERecHits",edm::InputTag("reducedEcalRecHitsEE"));
  desc.add<edm::InputTag>("rho",edm::InputTag("fixedGridRhoFastjetAllTmp"));
  desc.add<edm::InputTag>("BeamSpot",edm::InputTag("offlineBeamSpot"));
  desc.add<std::string>("Fitter","GsfTrajectoryFitter_forPreId");
  desc.add<std::string>("Smoother","GsfTrajectorySmoother_forPreId");
  desc.add<std::string>("TTRHBuilder","WithAngleAndTemplate");
  desc.add< std::vector<std::string> >("ModelNames",std::vector<std::string>());
  desc.add< std::vector<std::string> >("ModelWeights",std::vector<std::string>());
  desc.add< std::vector<double> >("ModelThrsholds",std::vector<double>());
  desc.add<bool>("PassThrough",false);
  desc.add<bool>("UsePfTracks",false);
  desc.add<double>("MinPtThreshold",0.5);
  desc.add<double>("MaxPtThreshold",15.);
}
