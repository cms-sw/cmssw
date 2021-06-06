// -*- C++ -*-
//
// Package:    Calibration/TkAlCaRecoProducers
// Class:      NearbyPixelClustersAnalyzer
//
/**\class NearbyPixelClustersAnalyzer NearbyPixelClustersAnalyzer.cc Calibration/TkAlCaRecoProducers/plugins/NearbyPixelClustersAnalyzer.cc

 Description: Analysis of the closebyPixelClusters collections
*/
//
// Original Author:  Marco Musich
//         Created:  Tue, 30 Mar 2021 09:22:07 GMT
//
//

// system include files
#include <memory>

// user include files
#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include "CalibTracker/SiPixelESProducers/interface/SiPixelDetInfoFileReader.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/Common/interface/DetSetVectorNew.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/SiPixelCluster/interface/SiPixelCluster.h"
#include "DataFormats/SiPixelDetId/interface/PixelSubdetector.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrackerCommon/interface/PixelBarrelName.h"
#include "DataFormats/TrackerCommon/interface/PixelEndcapName.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "DataFormats/TrackerRecHit2D/interface/SiPixelRecHit.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/StreamID.h"
#include "Geometry/CommonDetUnit/interface/PixelGeomDetUnit.h"
#include "Geometry/CommonTopologies/interface/PixelTopology.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/TrackerNumberingBuilder/interface/GeometricDet.h"
#include "RecoLocalTracker/ClusterParameterEstimator/interface/PixelClusterParameterEstimator.h"
#include "RecoLocalTracker/Records/interface/TkPixelCPERecord.h"
#include "TrackingTools/PatternTools/interface/TrajTrackAssociation.h"
#include "TrackingTools/TrackFitters/interface/TrajectoryStateCombiner.h"
#include "TrackingTools/TrajectoryState/interface/FreeTrajectoryState.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"
#include "TrackingTools/TransientTrackingRecHit/interface/TransientTrackingRecHit.h"

#include "TH2F.h"
//
// class declaration
//

class NearbyPixelClustersAnalyzer : public edm::one::EDAnalyzer<edm::one::SharedResources, edm::one::WatchRuns> {
public:
  explicit NearbyPixelClustersAnalyzer(const edm::ParameterSet&);
  ~NearbyPixelClustersAnalyzer() override = default;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void beginJob() override;
  void analyze(const edm::Event&, const edm::EventSetup&) override;
  void beginRun(edm::Run const&, edm::EventSetup const&) override;
  void endRun(edm::Run const&, edm::EventSetup const&) override{};
  std::map<uint32_t, TH2F*> bookModuleHistograms(const TrackerTopology* tTopo);
  std::tuple<std::string, int, int, int> setTopoInfo(uint32_t detId, const TrackerTopology* tTopo);
  void endJob() override;

  void countClusters(const edm::Handle<SiPixelClusterCollectionNew>& handle,
                     //const TrackerGeometry* tkGeom_,
                     unsigned int& nClusGlobal);

  bool detidIsOnPixel(const DetId& detid);
  TrajectoryStateOnSurface getTrajectoryStateOnSurface(const TrajectoryMeasurement& measurement);
  std::pair<float, float> findClosestCluster(const edm::Handle<SiPixelClusterCollectionNew>& handle,
                                             const PixelClusterParameterEstimator* pixelCPE_,
                                             const TrackerGeometry* trackerGeometry_,
                                             uint32_t rawId,
                                             float traj_lx,
                                             float traj_ly);

  void fillClusterFrames(const edm::Handle<SiPixelClusterCollectionNew>& handle);

  // ----------member data ---------------------------
  const edm::ESGetToken<TrackerGeometry, TrackerDigiGeometryRecord> geomEsToken_;
  const edm::ESGetToken<PixelClusterParameterEstimator, TkPixelCPERecord> pixelCPEEsToken_;
  const edm::ESGetToken<TrackerGeometry, TrackerDigiGeometryRecord> geomEsTokenBR_;
  const edm::ESGetToken<TrackerTopology, TrackerTopologyRcd> topoEsTokenBR_;

  edm::EDGetTokenT<SiPixelClusterCollectionNew> clustersToken_;
  edm::EDGetTokenT<SiPixelClusterCollectionNew> nearByClustersToken_;
  edm::EDGetTokenT<TrajTrackAssociationCollection> trajTrackCollectionToken_;
  edm::EDGetTokenT<edm::ValueMap<std::vector<float>>> distanceToken_;
  edm::EDGetTokenT<edm::View<reco::Track>> muonTracksToken_;

  edm::Service<TFileService> fs;

  TH1I* h_nALCARECOClusters;
  TH1I* h_nCloseByClusters;
  TH1F* h_distClosestValid;
  TH1F* h_distClosestMissing;
  TH1F* h_distClosestInactive;
  TH1F* h_distClosestTrack;

  SiPixelDetInfoFileReader reader_;
  std::map<std::string, TFileDirectory> outputFolders_;
  std::map<uint32_t, TH2F*> histoMap_;
  bool phase_;
};

//
// constructors and destructor
//
NearbyPixelClustersAnalyzer::NearbyPixelClustersAnalyzer(const edm::ParameterSet& iConfig)
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
      reader_(edm::FileInPath(iConfig.getParameter<std::string>("skimmedGeometryPath")).fullPath()) {
  usesResource(TFileService::kSharedResource);
}

//
// member functions
//

// ------------ method called for each event  ------------
void NearbyPixelClustersAnalyzer::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
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

    LogDebug("NearbyPixelClustersAnalyzer") << "distances size: " << distances.size() << std::endl;

    unsigned counter = 0;
    double closestDR = 999.;
    for (const auto& distance : distances) {
      counter++;
      LogDebug("NearbyPixelClustersAnalyzer")
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
  edm::LogInfo("NearbyPixelClustersAnalyzer") << "total ALCARECO clusters: " << nClusGlobal << std::endl;

  // Get nearby cluster collection
  const auto& nearByClusterCollectionHandle = iEvent.getHandle(nearByClustersToken_);

  unsigned int nNearByClusGlobal = 0;
  countClusters(nearByClusterCollectionHandle, /*trackerGeometry_,*/ nNearByClusGlobal);

  h_nCloseByClusters->Fill(nNearByClusGlobal);
  edm::LogInfo("NearbyPixelClustersAnalyzer") << "total close-by clusters: " << nNearByClusGlobal << std::endl;

  // fill the detector frames
  fillClusterFrames(nearByClusterCollectionHandle);

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
        edm::LogInfo("NearbyPixelClustersAnalyzer")
            << "RawID:" << r_rawId.rawId() << " (valid hit), distance: " << dist << std::endl;
        h_distClosestValid->Fill(dist);
      }

      if (recHit->getType() == TrackingRecHit::missing) {
        edm::LogInfo("NearbyPixelClustersAnalyzer")
            << "RawID:" << r_rawId.rawId() << " (missing hit), distance: " << dist << std::endl;
        h_distClosestMissing->Fill(dist);
      }

      if (recHit->getType() == TrackingRecHit::inactive) {
        edm::LogInfo("NearbyPixelClustersAnalyzer")
            << "RawID:" << r_rawId.rawId() << " (inactive hit), distance: " << dist << std::endl;
        h_distClosestInactive->Fill(dist);
      }
    }
  }
}

// ------------ method called once each job just before starting event loop  ------------
void NearbyPixelClustersAnalyzer::beginJob() {
  TFileDirectory ClusterCounts = fs->mkdir("ClusterCounts");
  h_nALCARECOClusters = ClusterCounts.make<TH1I>(
      "h_nALCARECOClusters", "Number of Pixel clusters per event (ALCARECO) ;N_{clusters};events", 20, 0, 20);
  h_nCloseByClusters = ClusterCounts.make<TH1I>(
      "h_nCloseByClusters", "Number of Pixel clusters per event (close-by) ;N_{clusters};events", 20, 0, 20);

  TFileDirectory Distances = fs->mkdir("TrajDistance");
  h_distClosestValid = Distances.make<TH1F>(
      "h_distClosestValid",
      "Distance of Closest cluster to trajectory (valid);distance (cm); valid trajectory measurements",
      110,
      -0.105,
      0.995);
  h_distClosestMissing = Distances.make<TH1F>(
      "h_distClosestMissing",
      "Distance of Closest cluster to trajectory (missing);distance (cm);missing trajectory measurements",
      110,
      -0.105,
      0.995);
  h_distClosestInactive = Distances.make<TH1F>(
      "h_distClosestInactive",
      "Distance of Closest cluster to trajectory (inactive);distance (cm);inactive trajectory measurements",
      110,
      -0.105,
      0.995);

  TFileDirectory TkDistances = fs->mkdir("OtherTrackDistance");
  h_distClosestTrack = TkDistances.make<TH1F>(
      "h_distClosestTrack",
      "#DeltaR Distance of Closest track to the muon trajectory;#DeltaR distance; muon trajectories",
      100,
      0.,
      5.);
}

// ------------ method called once each job just after ending the event loop  ------------
void NearbyPixelClustersAnalyzer::endJob() {
  // please remove this method if not needed
}

/*--------------------------------------------------------------------*/
bool NearbyPixelClustersAnalyzer::detidIsOnPixel(const DetId& detid)
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
TrajectoryStateOnSurface NearbyPixelClustersAnalyzer::getTrajectoryStateOnSurface(
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
void NearbyPixelClustersAnalyzer::countClusters(const edm::Handle<SiPixelClusterCollectionNew>& handle,
                                                //const TrackerGeometry* tkGeom_,
                                                unsigned int& nClusGlobal)
/*--------------------------------------------------------------------*/
{
  for (const auto& DSVItr : *handle) {
    uint32_t rawid(DSVItr.detId());
    DetId detId(rawid);
    LogDebug("NearbyPixelClustersAnalyzer") << "DetId: " << detId.rawId() << " size: " << DSVItr.size() << std::endl;
    nClusGlobal += DSVItr.size();
  }
}

/*--------------------------------------------------------------------*/
std::pair<float, float> NearbyPixelClustersAnalyzer::findClosestCluster(
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

  float minD = 10000.;

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

      float D = sqrt((lp.x() - traj_lx) * (lp.x() - traj_lx) + (lp.y() - traj_ly) * (lp.y() - traj_ly));
      if (D < minD) {
        minD = D;
        loc.first = (lp.x() - traj_lx);
        loc.second = (lp.y() - traj_ly);
      }
    }  // loop on cluster sets
  }
  return loc;
}

void NearbyPixelClustersAnalyzer::fillClusterFrames(const edm::Handle<SiPixelClusterCollectionNew>& handle) {
  const SiPixelClusterCollectionNew& clusterCollection = *handle;
  SiPixelClusterCollectionNew::const_iterator itClusterSet = clusterCollection.begin();

  for (; itClusterSet != clusterCollection.end(); itClusterSet++) {
    DetId detId(itClusterSet->id());

    edmNew::DetSet<SiPixelCluster>::const_iterator itCluster = itClusterSet->begin();
    for (; itCluster != itClusterSet->end(); ++itCluster) {
      const std::vector<SiPixelCluster::Pixel> pixelsVec = (*itCluster).pixels();
      for (unsigned int i = 0; i < pixelsVec.size(); ++i) {
        float pixx = pixelsVec[i].x;  // index as float=iteger, row index
        float pixy = pixelsVec[i].y;  // same, col index
        float pixel_charge = pixelsVec[i].adc;
        histoMap_[detId.rawId()]->Fill(pixy, pixx, pixel_charge);
      }
    }
  }
}

// ------------ method called for each run  ------------------------------------------
void NearbyPixelClustersAnalyzer::beginRun(const edm::Run& iRun, edm::EventSetup const& iSetup)
/*-----------------------------------------------------------------------------------*/
{
  edm::LogInfo("NearbyPixelClustersAnalyzer")
      << "@SUB=NearbyPixelClustersAnalyzer::beginRun() before booking histoMap_.size(): " << histoMap_.size()
      << std::endl;

  const TrackerTopology* tTopo_ = &iSetup.getData(topoEsTokenBR_);
  const TrackerGeometry* pDD_ = &iSetup.getData(geomEsTokenBR_);

  if ((pDD_->isThere(GeomDetEnumerators::P1PXB)) || (pDD_->isThere(GeomDetEnumerators::P1PXEC))) {
    phase_ = true;
  } else {
    phase_ = false;
  }

  unsigned nPixelDets = 0;
  for (const auto& it : pDD_->detUnits()) {
    const PixelGeomDetUnit* mit = dynamic_cast<PixelGeomDetUnit const*>(it);
    if (mit != nullptr) {
      nPixelDets++;
    }
  }

  const auto& detIds = reader_.getAllDetIds();
  if (detIds.size() != nPixelDets) {
    throw cms::Exception("Inconsistent Data")
        << "The size of the detId list specified from file (" << detIds.size()
        << ") differs from the one in TrackerGeometry (" << nPixelDets << ")! Please cross-check" << std::endl;
  }

  for (const auto& it : detIds) {
    auto topolInfo = setTopoInfo(it, tTopo_);

    std::string thePart = std::get<0>(topolInfo);

    // book the TFileDirectory if it's not already done
    if (!outputFolders_.count(thePart)) {
      LogDebug("NearbyPixelClustersAnalyzer") << "booking " << thePart << std::endl;
      outputFolders_[thePart] = fs->mkdir(thePart);
    }
  }

  if (histoMap_.empty()) {
    histoMap_ = bookModuleHistograms(tTopo_);
  }

  edm::LogInfo("NearbyPixelClusterAnalyzer")
      << "@SUB=NearbyPixelClusterAnalyzer::beginRun() After booking histoMap_.size(): " << histoMap_.size()
      << std::endl;
}

// ------------ method called to determine the topology  ------------
std::tuple<std::string, int, int, int> NearbyPixelClustersAnalyzer::setTopoInfo(uint32_t detId,
                                                                                const TrackerTopology* tTopo)
/*-----------------------------------------------------------------------------------*/
{
  int subdetId_(-999), layer_(-999), side_(-999);
  std::string ret = "";

  subdetId_ = DetId(detId).subdetId();
  switch (subdetId_) {
    case PixelSubdetector::PixelBarrel:  // PXB
      layer_ = tTopo->pxbLayer(detId);
      side_ = 0;
      ret += Form("BPix_Layer%i", layer_);
      break;
    case PixelSubdetector::PixelEndcap:  //PXF
      side_ = tTopo->pxfSide(detId);
      layer_ = tTopo->pxfDisk(detId);
      ret += ("FPix_");
      ret += (side_ == 1) ? Form("P_disk%i", layer_) : Form("M_disk%i", layer_);
      break;
    default:
      edm::LogError("NearbyPixelClusterAnalyzer") << "we should never be here!" << std::endl;
      break;
  }

  return std::make_tuple(ret, subdetId_, layer_, side_);
}

/* ------------ method called once to book all the module level histograms  ---------*/
std::map<uint32_t, TH2F*> NearbyPixelClustersAnalyzer::bookModuleHistograms(const TrackerTopology* tTopo_)
/*-----------------------------------------------------------------------------------*/
{
  std::map<uint32_t, TH2F*> hd;

  const auto& detIds = reader_.getAllDetIds();
  for (const auto& it : detIds) {
    // check if det id is correct and if it is actually cabled in the detector
    if (it == 0 || it == 0xFFFFFFFF) {
      edm::LogError("DetIdNotGood") << "@SUB=analyze"
                                    << "Wrong det id: " << it << "  ... neglecting!" << std::endl;
      continue;
    }

    auto topolInfo = setTopoInfo(it, tTopo_);
    std::string thePart = std::get<0>(topolInfo);

    unsigned int nCols = reader_.getDetUnitDimensions(it).first;
    unsigned int nRows = reader_.getDetUnitDimensions(it).second;

    int subdetId = DetId(it).subdetId();

    std::string moduleName = (subdetId == PixelSubdetector::PixelBarrel) ? PixelBarrelName(it, tTopo_, phase_).name()
                                                                         : PixelEndcapName(it, tTopo_, phase_).name();

    hd[it] = outputFolders_[thePart].make<TH2F>(
        Form("ClusterFrame_%s", moduleName.c_str()),
        Form("Cluster Map for module %s;n. cols;n. rows;pixel charge [ADC counts]", moduleName.c_str()),
        nCols,
        0,
        nCols,
        nRows,
        0,
        nRows);
  }

  return hd;
}

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void NearbyPixelClustersAnalyzer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.setComment("Analysis of the closebyPixelClusters collections");
  desc.add<edm::InputTag>("clusterCollection", edm::InputTag("ALCARECOSiPixelCalSingleMuonTight"));
  desc.add<edm::InputTag>("nearByClusterCollection", edm::InputTag("closebyPixelClusters"));
  desc.add<edm::InputTag>("trajectoryInput", edm::InputTag("refittedTracks"));
  desc.add<edm::InputTag>("muonTracks", edm::InputTag("ALCARECOSiPixelCalSingleMuonTight"));
  desc.add<edm::InputTag>("distToTrack", edm::InputTag("trackDistances"));
  desc.add<std::string>("skimmedGeometryPath",
                        "SLHCUpgradeSimulations/Geometry/data/PhaseI/PixelSkimmedGeometry_phase1.txt");
  descriptions.addWithDefaultLabel(desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(NearbyPixelClustersAnalyzer);
