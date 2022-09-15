// -*- C++ -*-
//
// Package:    DQMOffline/CalibTracker
// Class:      SiPixelCalSingleMuonAnalyzer
//
/**\class SiPixelCalSingleMuonAnalyzer SiPixelCalSingleMuonAnalyzer.cc DQMOffline/CalibTracker/plugins/SiPixelCalSingleMuonAnalyzer.cc

 Description: Analysis of the closebyPixelClusters collections for Pixel Hit Efficiency mearurements purposes
*/
//
// Original Author:  Marco Musich
//         Created:  Tue, 30 Mar 2021 09:22:07 GMT
//
//

// system include files
#include <memory>

// user include files
#include "CalibTracker/SiPixelESProducers/interface/SiPixelDetInfoFileReader.h"
#include "DQMServices/Core/interface/DQMEDAnalyzer.h"
#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/Common/interface/DetSetVectorNew.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/SiPixelCluster/interface/SiPixelCluster.h"
#include "DataFormats/SiPixelDetId/interface/PixelSubdetector.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "DataFormats/TrackerRecHit2D/interface/SiPixelRecHit.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "Geometry/CommonDetUnit/interface/PixelGeomDetUnit.h"
#include "Geometry/CommonTopologies/interface/PixelTopology.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "RecoLocalTracker/ClusterParameterEstimator/interface/PixelClusterParameterEstimator.h"
#include "RecoLocalTracker/Records/interface/TkPixelCPERecord.h"
#include "TrackingTools/PatternTools/interface/TrajTrackAssociation.h"
#include "TrackingTools/TrackFitters/interface/TrajectoryStateCombiner.h"
#include "TrackingTools/TrajectoryState/interface/FreeTrajectoryState.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"
#include "TrackingTools/TransientTrackingRecHit/interface/TransientTrackingRecHit.h"

//
// class declaration
//
class SiPixelCalSingleMuonAnalyzer : public DQMEDAnalyzer {
public:
  explicit SiPixelCalSingleMuonAnalyzer(const edm::ParameterSet&);
  ~SiPixelCalSingleMuonAnalyzer() override = default;
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void bookHistograms(DQMStore::IBooker&, edm::Run const&, edm::EventSetup const&) override;
  void analyze(const edm::Event&, const edm::EventSetup&) override;
  void countClusters(const edm::Handle<SiPixelClusterCollectionNew>& handle, unsigned int& nClusGlobal);
  const bool detidIsOnPixel(const DetId& detid);

  TrajectoryStateOnSurface getTrajectoryStateOnSurface(const TrajectoryMeasurement& measurement);
  std::pair<float, float> findClosestCluster(const edm::Handle<SiPixelClusterCollectionNew>& handle,
                                             const PixelClusterParameterEstimator* pixelCPE_,
                                             const TrackerGeometry* trackerGeometry_,
                                             uint32_t rawId,
                                             float traj_lx,
                                             float traj_ly);

  // ----------member data ---------------------------
  const edm::ESGetToken<TrackerGeometry, TrackerDigiGeometryRecord> geomEsToken_;
  const edm::ESGetToken<PixelClusterParameterEstimator, TkPixelCPERecord> pixelCPEEsToken_;
  const edm::ESGetToken<TrackerGeometry, TrackerDigiGeometryRecord> geomEsTokenBR_;
  const edm::ESGetToken<TrackerTopology, TrackerTopologyRcd> topoEsTokenBR_;

  const edm::EDGetTokenT<SiPixelClusterCollectionNew> clustersToken_;
  const edm::EDGetTokenT<SiPixelClusterCollectionNew> nearByClustersToken_;
  const edm::EDGetTokenT<TrajTrackAssociationCollection> trajTrackCollectionToken_;
  const edm::EDGetTokenT<edm::ValueMap<std::vector<float>>> distanceToken_;
  const edm::EDGetTokenT<edm::View<reco::Track>> muonTracksToken_;

  const std::string dqm_path_;
  SiPixelDetInfoFileReader reader_;

  typedef dqm::reco::MonitorElement MonitorElement;
  MonitorElement* h_nALCARECOClusters;
  MonitorElement* h_nCloseByClusters;
  MonitorElement* h_distClosestValid;
  MonitorElement* h_distClosestMissing;
  MonitorElement* h_distClosestInactive;
  MonitorElement* h_distClosestTrack;
  bool phase_;
};

//
// constructors and destructor
//
SiPixelCalSingleMuonAnalyzer::SiPixelCalSingleMuonAnalyzer(const edm::ParameterSet& iConfig)
    : geomEsToken_(esConsumes()),
      pixelCPEEsToken_(esConsumes(edm::ESInputTag("", "PixelCPEGeneric"))),
      geomEsTokenBR_(esConsumes<edm::Transition::BeginRun>()),
      topoEsTokenBR_(esConsumes<edm::Transition::BeginRun>()),
      clustersToken_(consumes<SiPixelClusterCollectionNew>(iConfig.getParameter<edm::InputTag>("clusterCollection"))),
      nearByClustersToken_(
          consumes<SiPixelClusterCollectionNew>(iConfig.getParameter<edm::InputTag>("nearByClusterCollection"))),
      trajTrackCollectionToken_(
          consumes<TrajTrackAssociationCollection>(iConfig.getParameter<edm::InputTag>("trajectoryInput"))),
      distanceToken_(consumes<edm::ValueMap<std::vector<float>>>(iConfig.getParameter<edm::InputTag>("distToTrack"))),
      muonTracksToken_(consumes<edm::View<reco::Track>>(iConfig.getParameter<edm::InputTag>("muonTracks"))),
      dqm_path_(iConfig.getParameter<std::string>("dqmPath")),
      reader_(edm::FileInPath(iConfig.getParameter<std::string>("skimmedGeometryPath")).fullPath()) {}

//
// member functions
//

// ------------ method called for each event  ------------
void SiPixelCalSingleMuonAnalyzer::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  using namespace edm;

  // get the Tracker geometry from event setup
  const TrackerGeometry* trackerGeometry_ = &iSetup.getData(geomEsToken_);

  // get the Pixel CPE from event setup
  const PixelClusterParameterEstimator* pixelCPE_ = &iSetup.getData(pixelCPEEsToken_);

  // get the muon track collection
  const auto& muonTrackCollectionHandle = iEvent.getHandle(muonTracksToken_);
  auto const& muonTracks = *muonTrackCollectionHandle;

  // get the track distances
  const auto& distancesToTrack = iEvent.getHandle(distanceToken_);

  unsigned int nMuons = muonTracks.size();
  for (unsigned int ij = 0; ij < nMuons; ij++) {
    auto muon = muonTrackCollectionHandle->ptrAt(ij);
    edm::RefToBase<reco::Track> trackRef = muonTrackCollectionHandle->refAt(ij);
    const auto& distances = (*distancesToTrack)[trackRef];

    LogDebug("SiPixelCalSingleMuonAnalyzer") << "distances size: " << distances.size() << std::endl;

    unsigned counter = 0;
    double closestDR = 999.;
    for (const auto& distance : distances) {
      counter++;
      LogDebug("SiPixelCalSingleMuonAnalyzer")
          << "track: " << counter << " distance:" << std::sqrt(distance) << std::endl;
      if (distance < closestDR && distance > 0) {
        closestDR = distance;
      }
    }

    h_distClosestTrack->Fill(std::sqrt(closestDR));
  }

  // Get cluster collection
  const auto& clusterCollectionHandle = iEvent.getHandle(clustersToken_);

  unsigned int nClusGlobal = 0;
  countClusters(clusterCollectionHandle, nClusGlobal);

  h_nALCARECOClusters->Fill(nClusGlobal);
  LogTrace("SiPixelCalSingleMuonAnalyzer") << "total ALCARECO clusters: " << nClusGlobal << std::endl;

  // Get nearby cluster collection
  const auto& nearByClusterCollectionHandle = iEvent.getHandle(nearByClustersToken_);

  unsigned int nNearByClusGlobal = 0;
  countClusters(nearByClusterCollectionHandle, nNearByClusGlobal);

  h_nCloseByClusters->Fill(nNearByClusGlobal);
  LogTrace("SiPixelCalSingleMuonAnalyzer") << "total close-by clusters: " << nNearByClusGlobal << std::endl;

  // Get Traj-Track Collection
  const auto& trajTrackCollectionHandle = iEvent.getHandle(trajTrackCollectionToken_);

  if (!trajTrackCollectionHandle.isValid())
    return;

  for (const auto& pair : *trajTrackCollectionHandle) {
    const edm::Ref<std::vector<Trajectory>> traj = pair.key;
    const reco::TrackRef track = pair.val;

    for (const TrajectoryMeasurement& measurement : traj->measurements()) {
      if (!measurement.updatedState().isValid())
        return;

      const TransientTrackingRecHit::ConstRecHitPointer& recHit = measurement.recHit();

      // Only looking for pixel hits
      DetId r_rawId = recHit->geographicalId();

      if (!this->detidIsOnPixel(r_rawId))
        continue;

      // Skipping hits with undeterminable positions
      TrajectoryStateOnSurface trajStateOnSurface = this->getTrajectoryStateOnSurface(measurement);

      if (!(trajStateOnSurface.isValid()))
        continue;

      // Position measurements
      // Looking for valid and missing hits
      LocalPoint localPosition = trajStateOnSurface.localPosition();

      const auto& traj_lx = localPosition.x();
      const auto& traj_ly = localPosition.y();

      const auto loc = this->findClosestCluster(
          nearByClusterCollectionHandle, pixelCPE_, trackerGeometry_, r_rawId.rawId(), traj_lx, traj_ly);

      float dist = (loc.first != -999.) ? std::sqrt(loc.first * loc.first + loc.second * loc.second) : -0.1;

      if (recHit->getType() == TrackingRecHit::valid) {
        LogTrace("SiPixelCalSingleMuonAnalyzer")
            << "RawID:" << r_rawId.rawId() << " (valid hit), distance: " << dist << std::endl;
        h_distClosestValid->Fill(dist);
      }

      if (recHit->getType() == TrackingRecHit::missing) {
        LogTrace("SiPixelCalSingleMuonAnalyzer")
            << "RawID:" << r_rawId.rawId() << " (missing hit), distance: " << dist << std::endl;
        h_distClosestMissing->Fill(dist);
      }

      if (recHit->getType() == TrackingRecHit::inactive) {
        LogTrace("SiPixelCalSingleMuonAnalyzer")
            << "RawID:" << r_rawId.rawId() << " (inactive hit), distance: " << dist << std::endl;
        h_distClosestInactive->Fill(dist);
      }
    }
  }
}

// ------------ method called once each job just before starting event loop  ------------
void SiPixelCalSingleMuonAnalyzer::bookHistograms(DQMStore::IBooker& iBooker, edm::Run const&, edm::EventSetup const&) {
  // book the overall event count and event types histograms
  iBooker.setCurrentFolder(dqm_path_ + "/ClusterCounts");
  h_nALCARECOClusters = iBooker.book1I(
      "h_nALCARECOClusters", "Number of Pixel clusters per event (ALCARECO) ;N_{clusters};events", 20, 0, 20);
  h_nCloseByClusters = iBooker.book1I(
      "h_nCloseByClusters", "Number of Pixel clusters per event (close-by) ;N_{clusters};events", 20, 0, 20);

  iBooker.setCurrentFolder(dqm_path_ + "/TrajDistance");
  h_distClosestValid =
      iBooker.book1D("h_distClosestValid",
                     "Distance of Closest cluster to trajectory (valid);distance (cm); valid trajectory measurements",
                     110,
                     -0.105,
                     0.995);
  h_distClosestMissing = iBooker.book1D(
      "h_distClosestMissing",
      "Distance of Closest cluster to trajectory (missing);distance (cm);missing trajectory measurements",
      110,
      -0.105,
      0.995);
  h_distClosestInactive = iBooker.book1D(
      "h_distClosestInactive",
      "Distance of Closest cluster to trajectory (inactive);distance (cm);inactive trajectory measurements",
      110,
      -0.105,
      0.995);

  iBooker.setCurrentFolder(dqm_path_ + "/OtherTrackDistance");
  h_distClosestTrack =
      iBooker.book1D("h_distClosestTrack",
                     "#DeltaR Distance of Closest track to the muon trajectory;#DeltaR distance; muon trajectories",
                     100,
                     0.,
                     5.);
}

/*--------------------------------------------------------------------*/
const bool SiPixelCalSingleMuonAnalyzer::detidIsOnPixel(const DetId& detid)
/*--------------------------------------------------------------------*/
{
  if (detid.det() != DetId::Tracker)
    return false;
  if (detid.subdetId() == PixelSubdetector::PixelBarrel)
    return true;
  if (detid.subdetId() == PixelSubdetector::PixelEndcap)
    return true;
  return false;
}

/*--------------------------------------------------------------------*/
TrajectoryStateOnSurface SiPixelCalSingleMuonAnalyzer::getTrajectoryStateOnSurface(
    const TrajectoryMeasurement& measurement)
/*--------------------------------------------------------------------*/
{
  const static TrajectoryStateCombiner trajStateCombiner;

  const auto& forwardPredictedState = measurement.forwardPredictedState();
  const auto& backwardPredictedState = measurement.backwardPredictedState();

  if (forwardPredictedState.isValid() && backwardPredictedState.isValid())
    return trajStateCombiner(forwardPredictedState, backwardPredictedState);

  else if (backwardPredictedState.isValid())
    return backwardPredictedState;

  else if (forwardPredictedState.isValid())
    return forwardPredictedState;

  edm::LogError("NearbyPixelClusterProducer") << "Error saving traj. measurement data."
                                              << " Trajectory state on surface cannot be determined." << std::endl;

  return TrajectoryStateOnSurface();
}

/*--------------------------------------------------------------------*/
void SiPixelCalSingleMuonAnalyzer::countClusters(const edm::Handle<SiPixelClusterCollectionNew>& handle,
                                                 //const TrackerGeometry* tkGeom_,
                                                 unsigned int& nClusGlobal)
/*--------------------------------------------------------------------*/
{
  for (const auto& DSVItr : *handle) {
    uint32_t rawid(DSVItr.detId());
    DetId detId(rawid);
    LogDebug("SiPixelCalSingleMuonAnalyzer") << "DetId: " << detId.rawId() << " size: " << DSVItr.size() << std::endl;
    nClusGlobal += DSVItr.size();
  }
}

/*--------------------------------------------------------------------*/
std::pair<float, float> SiPixelCalSingleMuonAnalyzer::findClosestCluster(
    const edm::Handle<SiPixelClusterCollectionNew>& handle,
    const PixelClusterParameterEstimator* pixelCPE_,
    const TrackerGeometry* trackerGeometry_,
    uint32_t rawId,
    float traj_lx,
    float traj_ly)
/*--------------------------------------------------------------------*/
{
  const SiPixelClusterCollectionNew& clusterCollection = *handle;
  SiPixelClusterCollectionNew::const_iterator itClusterSet = clusterCollection.begin();

  float minD = 10e7;

  auto loc = std::make_pair(-999., -999.);

  for (; itClusterSet != clusterCollection.end(); itClusterSet++) {
    DetId detId(itClusterSet->id());
    if (detId.rawId() != rawId)
      continue;

    unsigned int subDetId = detId.subdetId();
    if (subDetId != PixelSubdetector::PixelBarrel && subDetId != PixelSubdetector::PixelEndcap) {
      edm::LogError("NearByPixelClustersAnalyzer")
          << "ERROR: not a pixel cluster!!!" << std::endl;  // should not happen
      continue;
    }

    const PixelGeomDetUnit* pixdet = (const PixelGeomDetUnit*)trackerGeometry_->idToDetUnit(detId);
    edmNew::DetSet<SiPixelCluster>::const_iterator itCluster = itClusterSet->begin();
    for (; itCluster != itClusterSet->end(); ++itCluster) {
      LocalPoint lp(itCluster->x(), itCluster->y(), 0.);
      PixelClusterParameterEstimator::ReturnType params = pixelCPE_->getParameters(*itCluster, *pixdet);
      lp = std::get<0>(params);

      float D = (lp.x() - traj_lx) * (lp.x() - traj_lx) + (lp.y() - traj_ly) * (lp.y() - traj_ly);
      if (D < minD) {
        minD = D;
        loc.first = (lp.x() - traj_lx);
        loc.second = (lp.y() - traj_ly);
      }
    }  // loop on cluster sets
  }
  return loc;
}

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void SiPixelCalSingleMuonAnalyzer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.setComment("Analysis of the closebyPixelClusters collections");
  desc.add<edm::InputTag>("clusterCollection", edm::InputTag("ALCARECOSiPixelCalSingleMuonTight"));
  desc.add<edm::InputTag>("nearByClusterCollection", edm::InputTag("closebyPixelClusters"));
  desc.add<edm::InputTag>("trajectoryInput", edm::InputTag("refittedTracks"));
  desc.add<edm::InputTag>("muonTracks", edm::InputTag("ALCARECOSiPixelCalSingleMuonTight"));
  desc.add<edm::InputTag>("distToTrack", edm::InputTag("trackDistances"));
  desc.add<std::string>("dqmPath", "SiPixelCalSingleMuonTight");
  desc.add<std::string>("skimmedGeometryPath",
                        "SLHCUpgradeSimulations/Geometry/data/PhaseI/PixelSkimmedGeometry_phase1.txt");
  descriptions.addWithDefaultLabel(desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(SiPixelCalSingleMuonAnalyzer);
