#ifndef LowPtGsfElectronSCProducer_h
#define LowPtGsfElectronSCProducer_h

#include "DataFormats/EgammaReco/interface/SuperClusterFwd.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrackFwd.h"
#include "DataFormats/ParticleFlowReco/interface/GsfPFRecTrack.h"
#include "DataFormats/ParticleFlowReco/interface/GsfPFRecTrackFwd.h"
#include "DataFormats/ParticleFlowReco/interface/PFRecTrack.h"
#include "DataFormats/ParticleFlowReco/interface/PFRecTrackFwd.h"
#include "DataFormats/ParticleFlowReco/interface/PFCluster.h"
#include "DataFormats/ParticleFlowReco/interface/PFClusterFwd.h"
#include "DataFormats/ParticleFlowReco/interface/PFTrajectoryPoint.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"

class LowPtGsfElectronSCProducer : public edm::stream::EDProducer<> {
  
 public:
  
  explicit LowPtGsfElectronSCProducer( const edm::ParameterSet& );

  ~LowPtGsfElectronSCProducer();

  void produce( edm::Event&, const edm::EventSetup& );

 private:

  typedef edm::Ptr<reco::PFCluster> PFClusterPtr;
  typedef reco::PFClusterCollection PFClusters;
  typedef reco::GsfPFRecTrackCollection GsfPFRecTracks;
  typedef reco::PFRecTrackCollection PFRecTracks;
  typedef reco::GsfTrackCollection GsfTracks;
  typedef reco::SuperClusterCollection SuperClusters;
  typedef reco::PFTrajectoryPoint::LayerType LayerType;
    
  reco::PFClusterRef closest_cluster( const reco::PFTrajectoryPoint& point,
				      const edm::Handle<PFClusters>& clusters,
				      std::vector<int>& matched );
  
  edm::EDGetTokenT<GsfPFRecTracks> gsfPfRecTracks_;
  edm::EDGetTokenT<PFRecTracks> pfRecTracks_;
  edm::EDGetTokenT<GsfTracks> gsfTracks_;
  edm::EDGetTokenT<PFClusters> ecalClusters_;
  edm::EDGetTokenT<PFClusters> hcalClusters_;

};

#endif // LowPtGsfElectronSCProducer_h
