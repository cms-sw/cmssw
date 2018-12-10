#ifndef RecoEgamma_EgammaElectronProducers_LowPtGsfElectronSeedProducer_h
#define RecoEgamma_EgammaElectronProducers_LowPtGsfElectronSeedProducer_h

#include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "DataFormats/EgammaReco/interface/ElectronSeed.h"
#include "DataFormats/EgammaReco/interface/ElectronSeedFwd.h"
#include "DataFormats/ParticleFlowReco/interface/PFCluster.h"
#include "DataFormats/ParticleFlowReco/interface/PFClusterFwd.h"
#include "DataFormats/ParticleFlowReco/interface/PFRecTrack.h"
#include "DataFormats/ParticleFlowReco/interface/PFRecTrackFwd.h"
#include "DataFormats/ParticleFlowReco/interface/PreId.h"
#include "DataFormats/ParticleFlowReco/interface/PreIdFwd.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "MagneticField/Engine/interface/MagneticField.h"
#include "RecoEcal/EgammaCoreTools/interface/EcalClusterLazyTools.h"
#include "RecoEgamma/EgammaElectronProducers/interface/LowPtGsfElectronSeedHeavyObjectCache.h"

class LowPtGsfElectronSeedProducer final : public edm::stream::EDProducer< edm::GlobalCache<lowptgsfeleseed::HeavyObjectCache> >
{
  
 public:
  
  explicit LowPtGsfElectronSeedProducer( const edm::ParameterSet&, 
					 const lowptgsfeleseed::HeavyObjectCache* );

  ~LowPtGsfElectronSeedProducer() override;
  
  static std::unique_ptr<lowptgsfeleseed::HeavyObjectCache> 
    initializeGlobalCache( const edm::ParameterSet& conf ) {
    return std::make_unique<lowptgsfeleseed::HeavyObjectCache>(lowptgsfeleseed::HeavyObjectCache(conf));
  }
  
  static void globalEndJob( lowptgsfeleseed::HeavyObjectCache const* ) {}
  
  void beginLuminosityBlock( edm::LuminosityBlock const&, 
			     edm::EventSetup const& ) override;

  void produce( edm::Event&, const edm::EventSetup& ) override;
  
  static void fillDescription( edm::ParameterSetDescription& );
  
 private: // member functions
  
  template <typename T> void loop( const edm::Handle< std::vector<T> >& handle,
				   edm::Handle<reco::PFClusterCollection>& ecalClusters,
				   edm::Handle<reco::PFClusterCollection>& hcalClusters,
				   reco::ElectronSeedCollection& seeds,
				   reco::PreIdCollection& ecalPreIds, 
				   reco::PreIdCollection& hcalPreIds,
				   edm::Event&,
				   const edm::EventSetup& );

  // Overloaded methods to retrieve reco::TrackRef

  reco::TrackRef getBaseRef( edm::Handle< std::vector<reco::Track> > handle, int idx ) const;
  reco::TrackRef getBaseRef( edm::Handle< std::vector<reco::PFRecTrack> > handle, int idx ) const;

  // Overloaded methods to populate PreIds (using PF or KF tracks)

  void propagateTrackToCalo( const reco::PFRecTrackRef& pfTrackRef,
			     const edm::Handle<reco::PFClusterCollection>& ecalClusters,
			     const edm::Handle<reco::PFClusterCollection>& hcalClusters,
			     std::vector<int>& matchedEcalClusters,
			     std::vector<int>& matchedHcalClusters,
			     reco::PreId& ecalPreId, 
			     reco::PreId& hcalPreId );
  
  void propagateTrackToCalo( const reco::PFRecTrackRef& pfTrackRef,
			     const edm::Handle<reco::PFClusterCollection>& clusters, 
			     std::vector<int>& matchedClusters,
			     reco::PreId& preId,
			     bool ecal );

  void propagateTrackToCalo( const reco::TrackRef& pfTrack,
			     const edm::Handle<reco::PFClusterCollection>& ecalClusters,
			     const edm::Handle<reco::PFClusterCollection>& hcalClusters,
			     std::vector<int>& matchedEcalClusters,
			     std::vector<int>& matchedHcalClusters,
			     reco::PreId& ecalPreId, 
			     reco::PreId& hcalPreId );

  // Overloaded methods to evaluate BDTs (using PF or KF tracks)

  bool decision( const reco::PFRecTrackRef& pfTrackRef,
		 reco::PreId& ecal, 
		 reco::PreId& hcal,
		 double rho,
		 const reco::BeamSpot& spot,
		 noZS::EcalClusterLazyTools& ecalTools );
  
  bool decision( const reco::TrackRef& kfTrackRef,
		 reco::PreId& ecal, 
		 reco::PreId& hcal,
		 double rho,
		 const reco::BeamSpot& spot,
		 noZS::EcalClusterLazyTools& ecalTools );

  // Perform lightweight GSF tracking
  bool lightGsfTracking( reco::PreId&,
			 const reco::TrackRef&,
			 const reco::ElectronSeed&,
			 const edm::EventSetup& );

 private: // member data
  
  edm::ESHandle<MagneticField> field_;
  edm::EDGetTokenT<reco::TrackCollection> kfTracks_;
  edm::EDGetTokenT<reco::PFRecTrackCollection> pfTracks_;
  const edm::EDGetTokenT<reco::PFClusterCollection> ecalClusters_;
  const edm::EDGetTokenT<reco::PFClusterCollection> hcalClusters_;
  const edm::EDGetTokenT<EcalRecHitCollection> ebRecHits_;
  const edm::EDGetTokenT<EcalRecHitCollection> eeRecHits_;
  const edm::EDGetTokenT<double> rho_;
  const edm::EDGetTokenT<reco::BeamSpot> beamSpot_;
  const std::string fitter_;
  const std::string smoother_;
  const std::string builder_;
  const bool passThrough_;
  const bool usePfTracks_;
  const double minPtThreshold_;
  const double maxPtThreshold_;

};

#endif // RecoEgamma_EgammaElectronProducers_LowPtGsfElectronSeedProducer_h
