#include "RecoEgamma/EgammaElectronProducers/plugins/LowPtGsfElectronSCProducer.h"
#include "RecoParticleFlow/PFClusterTools/interface/PFClusterWidthAlgo.h"
#include "DataFormats/Common/interface/RefToPtr.h"
#include "DataFormats/Common/interface/ValueMap.h"
#include "DataFormats/EgammaReco/interface/SuperCluster.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrack.h"
#include "DataFormats/Math/interface/deltaR.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include <iostream>

LowPtGsfElectronSCProducer::LowPtGsfElectronSCProducer( const edm::ParameterSet& cfg )
{
  produces<SuperClusters>();
  produces< edm::ValueMap<reco::SuperClusterRef> >();
  gsfPfRecTracks_ = consumes<GsfPFRecTracks>( cfg.getParameter<edm::InputTag>("gsfPfRecTracks") );
  ecalClusters_ = consumes<PFClusters>( cfg.getParameter<edm::InputTag>("ecalClusters") );
  hcalClusters_ = consumes<PFClusters>( cfg.getParameter<edm::InputTag>("hcalClusters") );
}

LowPtGsfElectronSCProducer::~LowPtGsfElectronSCProducer()
{}

void LowPtGsfElectronSCProducer::produce( edm::Event& event, const edm::EventSetup& setup )
{

  edm::Handle<PFClusters> ecalClusters;
  event.getByToken(ecalClusters_,ecalClusters);
  if ( !ecalClusters.isValid() ) { edm::LogError("Problem with ecalClusters handle"); }

  edm::Handle<PFClusters> hcalClusters;
  event.getByToken(hcalClusters_,hcalClusters);
  if ( !hcalClusters.isValid() ) { edm::LogError("Problem with hcalClusters handle"); }

  edm::Handle<GsfPFRecTracks> gsfPfRecTracks;
  event.getByToken(gsfPfRecTracks_,gsfPfRecTracks);
  if ( !gsfPfRecTracks.isValid() ) { edm::LogError("Problem with gsfPfRecTracks handle"); }

  // SuperCluster container and getRefBeforePut
  std::unique_ptr<SuperClusters> clusters( new SuperClusters() );
  clusters->reserve(gsfPfRecTracks->size());
  const reco::SuperClusterRefProd clusters_refprod = event.getRefBeforePut<SuperClusters>();

  // ValueMap container
  std::vector<reco::SuperClusterRef> clusters_valuemap;

  // Iterate through GsfPfRecTracks and create corresponding SuperClusters
  std::vector<int> ecal_matched;
  for ( size_t igsfpf = 0; igsfpf < gsfPfRecTracks->size(); ++igsfpf ) { 

    // Refs to GSF(PF) tracks
    reco::GsfPFRecTrackRef gsfpf(gsfPfRecTracks, igsfpf);
    reco::GsfTrackRef gsf = gsfpf->gsfTrackRef();

    // Find closest "seed cluster" to GSF track extrapolated to ECAL
    const reco::PFTrajectoryPoint& point1 = gsfpf->extrapolatedPoint(LayerType::ECALShowerMax);
    reco::PFClusterRef best_seed = closest_cluster( point1, ecalClusters, ecal_matched );

    std::vector<reco::PFClusterRef> clusteredRefs;
    if(!best_seed.isNull()) clusteredRefs.push_back(best_seed);

    // Iterate through brem trajectories
    const std::vector<reco::PFBrem>& brems = gsfpf->PFRecBrem();
    std::vector<reco::PFBrem>::const_iterator brem;
    for ( brem = brems.begin(); brem != brems.end(); ++brem ) {

      // Find closest "brem cluster" using brem trajectory extrapolated to ECAL
      const reco::PFTrajectoryPoint& point2 = brem->extrapolatedPoint(LayerType::ECALShowerMax);
      reco::PFClusterRef best_brem = closest_cluster( point2, ecalClusters, ecal_matched );

      if(!best_brem.isNull()) {
	if(best_seed.isNull()) best_seed = best_brem;
	clusteredRefs.push_back(best_brem);
      }

    }

    // If all else fails, attempt to extrapolate KF track and match to seed PF cluster
    reco::PFTrajectoryPoint point3;
    if ( best_seed.isNull() ) { 
      const reco::PFRecTrackRef& kfpf = gsfpf->kfPFRecTrackRef();
      point3 = kfpf->extrapolatedPoint(LayerType::ECALShowerMax);
      reco::PFClusterRef best_kf = closest_cluster( point3, ecalClusters, ecal_matched );
      if( !best_kf.isNull() ) { 
	best_seed = best_kf; 
	clusteredRefs.push_back(best_kf); 
      }
    }
    
    //    // get closest HCAL PF cluster
    //    auto hcal_position = pftrk->extrapolatedPoint(reco::PFTrajectoryPoint::HCALEntrance);
    //    PFClusterRef hcal_ref = closest_cluster(hcal_position, hcal_clusters);
    //    if(!hcal_ref.isNull()) {
    //      hcal_ktf_clusters_map.insert(std::pair<reco::TrackRef, PFClusterRef>(trk, hcal_ref));
    //    }

    //now we need to make the supercluster
    if(!best_seed.isNull()) {

      float posX=0.,posY=0.,posZ=0.;
      float scEnergy=0.;
      for(const auto clus : clusteredRefs){
	scEnergy+=clus->correctedEnergy();
	posX+=clus->correctedEnergy()*clus->position().X();
	posY+=clus->correctedEnergy()*clus->position().Y();
	posZ+=clus->correctedEnergy()*clus->position().Z();
      }
      posX/=scEnergy;
      posY/=scEnergy;
      posZ/=scEnergy;
      reco::SuperCluster new_sc(scEnergy,math::XYZPoint(posX,posY,posZ));   
      new_sc.setCorrectedEnergy(scEnergy);
      new_sc.setSeed(edm::refToPtr(best_seed));
      std::vector<const reco::PFCluster*> barePtrs;
      for(const auto clus : clusteredRefs){
	new_sc.addCluster(edm::refToPtr(clus));
	barePtrs.push_back(&*clus);
      }
      PFClusterWidthAlgo pfwidth(barePtrs);
      new_sc.setEtaWidth(pfwidth.pflowEtaWidth());
      new_sc.setPhiWidth(pfwidth.pflowPhiWidth());
      new_sc.rawEnergy();//cache the value of raw energy
      
      // Store new SuperCluster 
      clusters->push_back( new_sc );
      
      // Populate ValueMap container
      clusters_valuemap.push_back( reco::SuperClusterRef(clusters_refprod,igsfpf) );
    }else{
      clusters_valuemap.push_back( reco::SuperClusterRef(clusters_refprod.id()) );
    }
  }

  // Put SuperClusters in event
  event.put(std::move(clusters));

  // Put ValueMap<SuperClusterRef> in event
  std::unique_ptr< edm::ValueMap<reco::SuperClusterRef> > ptr( new edm::ValueMap<reco::SuperClusterRef>() );
  edm::ValueMap<reco::SuperClusterRef>::Filler filler(*ptr);
  filler.insert(gsfPfRecTracks, clusters_valuemap.begin(), clusters_valuemap.end());
  filler.fill();
  event.put(std::move(ptr));

}

reco::PFClusterRef LowPtGsfElectronSCProducer::closest_cluster( const reco::PFTrajectoryPoint& point,
								const edm::Handle<PFClusters>& clusters,
								std::vector<int>& matched ) {
  reco::PFClusterRef best_ref;
  if ( point.isValid() ) {
    float min_dr = 1.e6;
    for ( size_t ii = 0; ii < clusters->size(); ++ii ) {
      float dr = reco::deltaR( clusters->at(ii), point.positionREP() );
      if( dr < min_dr ) {// && std::find( matched.begin(), matched.end(), ii ) == matched.end() ) {//@@
	best_ref = reco::PFClusterRef( clusters, ii );
	min_dr = dr;
      }
    }
    //matched.push_back( best_ref.index() );//@@
  }
  return best_ref;
}
