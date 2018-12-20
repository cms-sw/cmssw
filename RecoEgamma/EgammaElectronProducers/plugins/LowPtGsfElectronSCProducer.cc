#include "RecoParticleFlow/PFClusterTools/interface/PFClusterWidthAlgo.h"
#include "DataFormats/Common/interface/RefToPtr.h"
#include "DataFormats/Common/interface/ValueMap.h"
#include "DataFormats/CaloRecHit/interface/CaloCluster.h"
#include "DataFormats/CaloRecHit/interface/CaloClusterFwd.h"
#include "DataFormats/EgammaReco/interface/SuperCluster.h"
#include "DataFormats/EgammaReco/interface/SuperClusterFwd.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrack.h"
#include "DataFormats/Math/interface/deltaR.h"
#include "DataFormats/ParticleFlowReco/interface/PFRecTrack.h"
#include "DataFormats/ParticleFlowReco/interface/PFRecTrackFwd.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "RecoEgamma/EgammaElectronProducers/plugins/LowPtGsfElectronSCProducer.h"
#include <iostream>

////////////////////////////////////////////////////////////////////////////////
//
LowPtGsfElectronSCProducer::LowPtGsfElectronSCProducer( const edm::ParameterSet& cfg ) :
  gsfPfRecTracks_{consumes<reco::GsfPFRecTrackCollection>( cfg.getParameter<edm::InputTag>("gsfPfRecTracks") )},
  ecalClusters_{consumes<reco::PFClusterCollection>( cfg.getParameter<edm::InputTag>("ecalClusters") )},
  dr2_{cfg.getParameter<double>("MaxDeltaR2")}
{
  produces<reco::CaloClusterCollection>();
  produces<reco::SuperClusterCollection>();
  produces< edm::ValueMap<reco::SuperClusterRef> >();
}

////////////////////////////////////////////////////////////////////////////////
//
LowPtGsfElectronSCProducer::~LowPtGsfElectronSCProducer()
{}

////////////////////////////////////////////////////////////////////////////////
//
void LowPtGsfElectronSCProducer::produce( edm::Event& event, const edm::EventSetup& setup )
{

  // Input GsfPFRecTracks collection
  edm::Handle<reco::GsfPFRecTrackCollection> gsfPfRecTracks;
  event.getByToken(gsfPfRecTracks_,gsfPfRecTracks);
  if ( !gsfPfRecTracks.isValid() ) { edm::LogError("Problem with gsfPfRecTracks handle"); }

  // Input EcalClusters collection
  edm::Handle<reco::PFClusterCollection> ecalClusters;
  event.getByToken(ecalClusters_,ecalClusters);
  if ( !ecalClusters.isValid() ) { edm::LogError("Problem with ecalClusters handle"); }

  // Output SuperClusters collection and getRefBeforePut
  auto superClusters = std::make_unique<reco::SuperClusterCollection>( reco::SuperClusterCollection() );
  superClusters->reserve(gsfPfRecTracks->size());
  const reco::SuperClusterRefProd superClustersRefProd = event.getRefBeforePut<reco::SuperClusterCollection>();

  // Output ValueMap container of GsfPFRecTrackRef index to SuperClusterRef
  std::vector<reco::SuperClusterRef> superClustersValueMap;

  // Output CaloClusters collection
  auto caloClusters = std::make_unique<reco::CaloClusterCollection>( reco::CaloClusterCollection() );
  caloClusters->reserve(ecalClusters->size());

  // Index[GSF track][trajectory point] for "best" CaloCluster
  std::vector< std::vector<int> > cluster_idx;
  cluster_idx.resize( gsfPfRecTracks->size(), std::vector<int>() );

  // Index[GSF track][trajectory point] for "best" PFCluster
  std::vector< std::vector<int> > pfcluster_idx;
  pfcluster_idx.resize( gsfPfRecTracks->size(), std::vector<int>() );

  // dr2min[GSF track][trajectory point] for "best" CaloCluster
  std::vector< std::vector<float> > cluster_dr2min;
  cluster_dr2min.resize( gsfPfRecTracks->size(), std::vector<float>() );

  // Construct list of trajectory points from the GSF track and electron brems
  std::vector< std::vector<const reco::PFTrajectoryPoint*> > points;
  points.resize( gsfPfRecTracks->size(), std::vector<const reco::PFTrajectoryPoint*>() );
  for ( size_t itrk = 0; itrk < gsfPfRecTracks->size(); ++itrk ) { 
    // Extrapolated track
    reco::GsfPFRecTrackRef trk(gsfPfRecTracks,itrk);
    points[itrk].reserve(trk->PFRecBrem().size()+1);
    points[itrk].push_back( &trk->extrapolatedPoint(reco::PFTrajectoryPoint::LayerType::ECALShowerMax) );
    // Extrapolated brem trajectories
    for ( auto brem : trk->PFRecBrem() ) {
      points[itrk].push_back( &brem.extrapolatedPoint(reco::PFTrajectoryPoint::LayerType::ECALShowerMax) ); 
    }
    // Resize containers
    cluster_idx[itrk].resize(points[itrk].size(),-1);
    pfcluster_idx[itrk].resize(points[itrk].size(),-1);
    cluster_dr2min[itrk].resize(points[itrk].size(),1.e6);
  }

  // For each cluster, find closest trajectory point ("global" arbitration at event level)
  for ( size_t iclu = 0; iclu < ecalClusters->size(); ++iclu ) { // Cluster loop
    std::pair<int,int> point = std::make_pair(-1,-1);
    float dr2min = 1.e6;
    for ( size_t ipoint = 0; ipoint < points.size(); ++ipoint ) { // GSF track loop
      for ( size_t jpoint = 0; jpoint < points[ipoint].size(); ++jpoint ) { // Traj point loop
	if ( points[ipoint][jpoint]->isValid() ) {
	  float dr2 = reco::deltaR2( ecalClusters->at(iclu), points[ipoint][jpoint]->positionREP() );
	  if ( dr2 < dr2min ) {
	    // Store nearest point to this cluster
	    dr2min = dr2;
	    point = std::make_pair(ipoint,jpoint);
	  }
	}
      }
    }
    if ( point.first >= 0 && point.second >= 0 && // if this cluster is matched to a point ...
	 dr2min < cluster_dr2min[point.first][point.second] ) { // ... and cluster is closest to the same point 
      // Copy CaloCluster to new collection
      caloClusters->push_back(ecalClusters->at(iclu));
      // Store cluster index for creation of SC later
      cluster_idx[point.first][point.second] = caloClusters->size()-1;
      pfcluster_idx[point.first][point.second] = iclu;
      cluster_dr2min[point.first][point.second] = dr2min;
    }
  }

  // Put CaloClusters in event and get orphan handle
  const edm::OrphanHandle<reco::CaloClusterCollection>& caloClustersH = event.put(std::move(caloClusters));

  // Loop through GSF tracks
  for ( size_t itrk = 0; itrk < gsfPfRecTracks->size(); ++itrk ) { 
    
    // Used to create SC
    float energy = 0.;
    float X = 0., Y = 0., Z = 0.;
    reco::CaloClusterPtr seed;
    reco::CaloClusterPtrVector clusters;
    std::vector<const reco::PFCluster*> barePtrs;

    // Find closest match in dr2 from points associated to given track
    int index = -1;
    float dr2 = 1.e6;
    for ( size_t ipoint = 0; ipoint < cluster_idx[itrk].size(); ++ipoint ) { 
      if ( cluster_idx[itrk][ipoint] < 0 ) { continue; }
      if ( cluster_dr2min[itrk][ipoint] < dr2 ) {
	dr2 = cluster_dr2min[itrk][ipoint];
	index = cluster_idx[itrk][ipoint];
      }
    }
 
    // For each track, loop through points and use associated cluster
    for ( size_t ipoint = 0; ipoint < cluster_idx[itrk].size(); ++ipoint ) { 
      if ( cluster_idx[itrk][ipoint] < 0 ) { continue; }
      reco::CaloClusterPtr clu(caloClustersH,cluster_idx[itrk][ipoint]);
      if ( clu.isNull() ) { continue; }
      if ( !( cluster_dr2min[itrk][ipoint] < dr2_ || // Require cluster to be closer than dr2_ ...
	      index == cluster_idx[itrk][ipoint] ) ) { continue; } // ... unless it is the closest one ...
      if ( seed.isNull() ) { seed = clu; }
      clusters.push_back(clu);
      energy += clu->correctedEnergy();
      X += clu->position().X() * clu->correctedEnergy();
      Y += clu->position().Y() * clu->correctedEnergy();
      Z += clu->position().Z() * clu->correctedEnergy();
      reco::PFClusterRef pfclu(ecalClusters,pfcluster_idx[itrk][ipoint]);
      if ( pfclu.isNonnull() ) { barePtrs.push_back(&*pfclu); }
    }
    X /= energy;
    Y /= energy;
    Z /= energy;

    // Create SC
    if ( seed.isNonnull() ) {
      reco::SuperCluster sc(energy,math::XYZPoint(X,Y,Z));
      sc.setCorrectedEnergy(energy);
      sc.setSeed(seed);
      sc.setClusters(clusters);
      for ( const auto clu : clusters ) { sc.addCluster(clu); }
      PFClusterWidthAlgo pfwidth(barePtrs);
      sc.setEtaWidth(pfwidth.pflowEtaWidth());
      sc.setPhiWidth(pfwidth.pflowPhiWidth());
      sc.rawEnergy(); // Cache the value of raw energy
      superClusters->push_back(sc);

      // Populate ValueMap container
      superClustersValueMap.push_back( reco::SuperClusterRef( superClustersRefProd, superClusters->size()-1 ) );
    } else {
      superClustersValueMap.push_back( reco::SuperClusterRef( superClustersRefProd.id() ) );
    }

  } // GSF tracks

  // Put SuperClusters in event
  event.put(std::move(superClusters));

  auto ptr = std::make_unique< edm::ValueMap<reco::SuperClusterRef> >( edm::ValueMap<reco::SuperClusterRef>() );
  edm::ValueMap<reco::SuperClusterRef>::Filler filler(*ptr);
  filler.insert(gsfPfRecTracks, superClustersValueMap.begin(), superClustersValueMap.end());
  filler.fill();
  event.put(std::move(ptr));

}

////////////////////////////////////////////////////////////////////////////////
//
reco::PFClusterRef LowPtGsfElectronSCProducer::closestCluster( const reco::PFTrajectoryPoint& point,
							       const edm::Handle<reco::PFClusterCollection>& clusters,
							       std::vector<int>& matched ) {
  reco::PFClusterRef closest;
  if ( point.isValid() ) {
    float dr2min = dr2_;
    for ( size_t ii = 0; ii < clusters->size(); ++ii ) {
      if ( std::find( matched.begin(), matched.end(), ii ) == matched.end() ) {
	float dr2 = reco::deltaR2( clusters->at(ii), point.positionREP() );
	if ( dr2 < dr2min ) {
	  closest = reco::PFClusterRef( clusters, ii );
	  dr2min = dr2;
	}
      }
    }
    if ( dr2min < (dr2_ - 1.e-6) ) { matched.push_back( closest.index() ); }
  }
  return closest;
}
 
//////////////////////////////////////////////////////////////////////////////////////////
//
void LowPtGsfElectronSCProducer::fillDescriptions( edm::ConfigurationDescriptions& descriptions )
{
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("gsfPfRecTracks",edm::InputTag("lowPtGsfElePfGsfTracks"));
  desc.add<edm::InputTag>("ecalClusters",edm::InputTag("particleFlowClusterECAL"));
  desc.add<edm::InputTag>("hcalClusters",edm::InputTag("particleFlowClusterHCAL"));
  desc.add<double>("MaxDeltaR2",1.);
  descriptions.add("lowPtGsfElectronSuperClusters",desc);
}
