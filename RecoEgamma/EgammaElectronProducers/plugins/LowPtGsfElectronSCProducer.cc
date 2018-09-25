#include "RecoEgamma/EgammaElectronProducers/plugins/LowPtGsfElectronSCProducer.h"
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
  pfRecTracks_ = consumes<PFRecTracks>( cfg.getParameter<edm::InputTag>("pfRecTracks") );
  gsfTracks_ = consumes<GsfTracks>( cfg.getParameter<edm::InputTag>("gsfTracks") );
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

  //@@ not necessary, debug only
  edm::Handle<PFRecTracks> pfRecTracks;
  event.getByToken(pfRecTracks_,pfRecTracks);
  if ( !pfRecTracks.isValid() ) { edm::LogError("Problem with pfRecTracks handle"); }

  //@@ not necessary, debug only
  edm::Handle<GsfTracks> gsfTracks;
  event.getByToken(gsfTracks_,gsfTracks);
  if ( !gsfTracks.isValid() ) { edm::LogError("Problem with gsfTracks handle"); }

  //@@ not necessary, debug only
//  std::cout << "[LowPtGsfElectronSCProducer::produce]"
//	    << " gsfPfRecTracks->size() " << gsfPfRecTracks->size()
//	    << " pfRecTracks->size() " << pfRecTracks->size()
//	    << " gsfTracks->size() " << gsfTracks->size()
//	    << std::endl;

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

    //@@ Only create SC if seed PFCluster found ... ???
    //if( best_seed.isNull() ) { continue; }

    // Create new SC, with energy and position based on (extrapolated) GSF track
    reco::SuperCluster new_sc( sqrt( gsf->p()*gsf->p() + 0.511E-3*0.511E-3 ), // energy
			       math::XYZPoint(point1.position().X(),
					      point1.position().Y(),
					      point1.position().Z()) );
    new_sc.setCorrectedEnergy( sqrt( gsf->p()*gsf->p() + 0.511E-3*0.511E-3 ) ); //@@ necessary?

    // Add "seed cluster" (matched to GSF track extrapolated to ECAL)
    if( !best_seed.isNull() ) { 
      new_sc.addCluster( edm::refToPtr(best_seed) ); 
      if ( new_sc.seed().isNull() ) { new_sc.setSeed( edm::refToPtr(best_seed) ); }
    }

    // Iterate through brem trajectories
    const std::vector<reco::PFBrem>& brems = gsfpf->PFRecBrem();
    std::vector<reco::PFBrem>::const_iterator brem;
    for ( brem = brems.begin(); brem != brems.end(); ++brem ) {

      // Find closest "brem cluster" using brem trajectory extrapolated to ECAL
      const reco::PFTrajectoryPoint& point2 = brem->extrapolatedPoint(LayerType::ECALShowerMax);
      reco::PFClusterRef best_brem = closest_cluster( point2, ecalClusters, ecal_matched );

      // Add brem cluster
      if( !best_brem.isNull() ) { 
	new_sc.addCluster( edm::refToPtr(best_brem) );
	if ( new_sc.seed().isNull() ) { new_sc.setSeed( edm::refToPtr(best_brem) ); }
      }

    }

    // If all else fails, attempt to extrapolate KF track and match to seed PF cluster
    reco::PFTrajectoryPoint point3;
    if ( new_sc.seed().isNull() ) { 
      const reco::PFRecTrackRef& kfpf = gsfpf->kfPFRecTrackRef();
      point3 = kfpf->extrapolatedPoint(LayerType::ECALShowerMax);
      reco::PFClusterRef best_kf = closest_cluster( point3, ecalClusters, ecal_matched );
      if( !best_kf.isNull() ) { 
	new_sc.addCluster( edm::refToPtr(best_kf) ); 
	new_sc.setSeed( edm::refToPtr(best_kf) ); 
      }
    }
    
    //    // get closest HCAL PF cluster
    //    auto hcal_position = pftrk->extrapolatedPoint(reco::PFTrajectoryPoint::HCALEntrance);
    //    PFClusterRef hcal_ref = closest_cluster(hcal_position, hcal_clusters);
    //    if(!hcal_ref.isNull()) {
    //      hcal_ktf_clusters_map.insert(std::pair<reco::TrackRef, PFClusterRef>(trk, hcal_ref));
    //    }

//    if ( new_sc.seed().isNull() ) { 
//      std::cout << " TEST " 
//		<< point1.isValid() << " "
//		<< point3.isValid() << " "
//		<< ( gsfpf->kfPFRecTrackRef()->trackRef().isNull() ? -1. : 
//		     gsfpf->kfPFRecTrackRef()->trackRef()->pt() ) << " "
//		<< ( gsfpf->gsfTrackRef().isNull() ? -1. : gsfpf->gsfTrackRef()->pt() ) << " "
//		<< ( gsfpf->gsfTrackRef().isNull() ? -1. : 
//		     sqrt(gsfpf->gsfTrackRef()->innerMomentum().Perp2()) ) << " "
//		<< ( gsfpf->gsfTrackRef().isNull() ? -1. : 
//		     sqrt(gsfpf->gsfTrackRef()->outerMomentum().Perp2()) ) << " "
//		<< std::endl;
//    }

    // Store new SuperCluster 
    clusters->push_back( new_sc );

    // Populate ValueMap container
    clusters_valuemap.push_back( reco::SuperClusterRef(clusters_refprod,igsfpf) );

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
  if( point.isValid() ) { 
    float min_dr = 9999.f;
    for( size_t ii = 0; ii < clusters->size(); ++ii ) {
      float dr = reco::deltaR( clusters->at(ii), point.positionREP() );
      if( dr < min_dr && std::find( matched.begin(), matched.end(), ii ) == matched.end() ) {
	best_ref = reco::PFClusterRef( clusters, ii );
	min_dr = dr;
      }
    }
    matched.push_back( best_ref.index() );
  }
  return best_ref;
}
