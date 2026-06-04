#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/EDGetToken.h"

#include "DataFormats/FTLRecHit/interface/FTLMergedClusterCollections.h"
#include "DataFormats/FTLRecHit/interface/FTLClusterCollections.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/ForwardDetId/interface/BTLDetId.h"
#include "DataFormats/ForwardDetId/interface/ETLDetId.h"

#include "Geometry/Records/interface/MTDDigiGeometryRecord.h"
#include "Geometry/Records/interface/MTDTopologyRcd.h"
#include "Geometry/MTDGeometryBuilder/interface/MTDGeometry.h"
#include "Geometry/MTDGeometryBuilder/interface/MTDTopology.h"
#include "Geometry/CommonDetUnit/interface/GeomDet.h"
#include "DataFormats/GeometryVector/interface/LocalPoint.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "DataFormats/GeometryCommonDetAlgo/interface/MeasurementPoint.h"
#include "Geometry/MTDGeometryBuilder/interface/RectangularMTDTopology.h"
#include "Geometry/MTDGeometryBuilder/interface/ProxyMTDTopology.h"

#include <iostream>
#include <vector>
#include <map>
#include <set>
#include <cmath>
#include <limits>

class MTDMergedClusterProducer : public edm::stream::EDProducer<> {
public:
  explicit MTDMergedClusterProducer(const edm::ParameterSet& conf);
  ~MTDMergedClusterProducer() override = default;

  void produce(edm::Event& e, const edm::EventSetup& es) override;
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  edm::EDGetTokenT<FTLClusterCollection> btlClustersToken_;
  edm::EDGetTokenT<FTLClusterCollection> etlClustersToken_;
  std::string btlMergedClusterInstance_;
  std::string etlMergedClusterInstance_;

  double timeThreshold_;
  double energyThreshold_;

  edm::ESGetToken<MTDGeometry, MTDDigiGeometryRecord> mtdgeoToken_;
  edm::ESGetToken<MTDTopology, MTDTopologyRcd> mtdtopoToken_;

  bool areTimingCompatible(const FTLCluster* c1, const FTLCluster* c2);
  FTLMergedCluster mergeClusters(const std::vector<const FTLCluster*>& clusters,
                                 const DetId& seedId,
                                 const MTDGeometry& geom,
                                 edm::Handle<FTLClusterCollection> btlClustersHandle);
};

MTDMergedClusterProducer::MTDMergedClusterProducer(const edm::ParameterSet& conf)
    : btlClustersToken_(consumes<FTLClusterCollection>(conf.getParameter<edm::InputTag>("btlBarrel"))),
      etlClustersToken_(consumes<FTLClusterCollection>(conf.getParameter<edm::InputTag>("etlEndcap"))),
      btlMergedClusterInstance_(conf.getParameter<std::string>("btlMergedClusterInstance")),
      etlMergedClusterInstance_(conf.getParameter<std::string>("etlMergedClusterInstance")),
      timeThreshold_(conf.getParameter<double>("timeThreshold")),
      energyThreshold_(conf.getParameter<double>("energyThreshold")),
      mtdgeoToken_(esConsumes<MTDGeometry, MTDDigiGeometryRecord>()),
      mtdtopoToken_(esConsumes<MTDTopology, MTDTopologyRcd>()) {
  produces<FTLMergedClusterCollection>(btlMergedClusterInstance_);
  produces<FTLMergedClusterCollection>(etlMergedClusterInstance_);
}

bool MTDMergedClusterProducer::areTimingCompatible(const FTLCluster* c1, const FTLCluster* c2) {
  double timeDiff = std::abs(c1->time() - c2->time());
  double timeError1 = c1->timeError();
  double timeError2 = c2->timeError();
  double combinedError = std::sqrt(timeError1 * timeError1 + timeError2 * timeError2);

  bool compatible = timeDiff < (timeThreshold_ * combinedError);

  return compatible;
}

FTLMergedCluster MTDMergedClusterProducer::mergeClusters(const std::vector<const FTLCluster*>& clusters,
                                                         const DetId& seedId,
                                                         const MTDGeometry& geom,
                                                         edm::Handle<FTLClusterCollection> mtdClustersHandle) {
  float totalEnergy = 0;
  float weightedTime = 0;
  float weightedTimeError2 = 0;

  double weightedGlobalX = 0;
  double weightedGlobalY = 0;
  double weightedGlobalZ = 0;

  std::vector<DetId> clusterIds;
  std::vector<FTLClusterRef> clusterRefs;

  // get primary cluster -- if only one, pick that detid, else pick the one with earliest time
  const FTLCluster* primary = clusters.front();
  if (clusters.size() > 1) {
    for (const auto* c : clusters) {
      if (c->time() < primary->time()) {
        primary = c;
      } else if (c->time() == primary->time() && c->energy() > primary->energy()) {
        primary = c;
      }
    }
  }

  // primary first, then the rest
  std::vector<const FTLCluster*> orderedClusters;
  orderedClusters.reserve(clusters.size());
  orderedClusters.push_back(primary);
  for (const auto* c : clusters) {
    if (c != primary)
      orderedClusters.push_back(c);
  }

  // energy-weighted average time and position in global coord:
  for (const auto* cluster : orderedClusters) {
    float energy = cluster->energy();

    totalEnergy += energy;
    weightedTime += energy * cluster->time();
    weightedTimeError2 += energy * energy * cluster->timeError() * cluster->timeError();

    clusterIds.push_back(cluster->id());
    clusterRefs.push_back(edmNew::makeRefTo(mtdClustersHandle, cluster));

    // convert clus pos to global coordinates
    const GeomDet* det = geom.idToDetUnit(cluster->id());
    if (det) {
      const ProxyMTDTopology& topoproxy = static_cast<const ProxyMTDTopology&>(det->topology());
      const RectangularMTDTopology& topo = static_cast<const RectangularMTDTopology&>(topoproxy.specificTopology());

      float localX = cluster->getClusterPosX();  // get from the cluster the position along crystal length (in cm)
      float localY = 0.0f;

      if (cluster->getClusterErrorX() < 0.) {  // in case it's not set, use topology
        MeasurementPoint mp(cluster->x(), cluster->y());
        const LocalPoint localPos = topo.localPosition(mp);
        localX = localPos.x();
        localY = localPos.y();
      } else {  // valid x from SiPM readout, but get y from topology (rows/cols)
        localY = topo.localY(cluster->y());
      }

      const LocalPoint localPos(localX, localY, 0.0f);
      const GlobalPoint gp = det->surface().toGlobal(localPos);
      weightedGlobalX += static_cast<double>(energy) * gp.x();
      weightedGlobalY += static_cast<double>(energy) * gp.y();
      weightedGlobalZ += static_cast<double>(energy) * gp.z();
    } else {
      edm::LogWarning("MTDMergedClusterProducer")
          << "Unable to convert cluster DetId " << cluster->id().rawId() << " to GlobalPoint, geometry not available";
    }
  }

  float avgTime = 0.f;
  float avgTimeError = 0.f;
  if (totalEnergy > 0) {
    avgTime = weightedTime / totalEnergy;
    avgTimeError = std::sqrt(weightedTimeError2) / totalEnergy;
  }

  GlobalPoint avgGlobal(0., 0., 0.);
  if (totalEnergy > 0) {
    avgGlobal =
        GlobalPoint(weightedGlobalX / totalEnergy, weightedGlobalY / totalEnergy, weightedGlobalZ / totalEnergy);
  }

  // now to seed-local
  float avgX = 0;
  float avgY = 0;
  const GeomDet* seedDet = geom.idToDetUnit(seedId);
  if (seedDet && totalEnergy > 0) {
    LocalPoint lp = seedDet->surface().toLocal(avgGlobal);
    avgX = lp.x();  // along crystal (phi)
    avgY = lp.y();  // perpendicular to crystal (eta)
  } else {
    edm::LogWarning("MTDMergedClusterProducer") << "Unable to convert avgGlobal to seed-local coordinates for seedId "
                                                << seedId.rawId() << " , geometry not available";
  }

  // errors:
  float avgXError = 0.f;
  float avgYError = 0.f;

  if (totalEnergy > 0) {
    double weightedErrorX2 = 0.;
    double weightedErrorY2 = 0.;

    for (const auto* cluster : clusters) {
      float energy = cluster->energy();
      const GeomDet* det = geom.idToDetUnit(cluster->id());
      if (!det)
        continue;

      const ProxyMTDTopology& topoproxy = static_cast<const ProxyMTDTopology&>(det->topology());
      const RectangularMTDTopology& topo = static_cast<const RectangularMTDTopology&>(topoproxy.specificTopology());

      float localXError = 0.f;
      float localYError = 0.f;

      if (cluster->getClusterErrorX() < 0.) {
        MeasurementPoint mp(cluster->x(), cluster->y());
        float sigma_flat = 1.0f / std::sqrt(12.0f);
        float sigma2 = cluster->positionError(sigma_flat);
        sigma2 *= sigma2;
        MeasurementError posErr(sigma2, 0, sigma2);
        LocalError localErr = topo.localError(mp, posErr);
        localXError = std::sqrt(localErr.xx());
        localYError = std::sqrt(localErr.yy());
      } else {  // use cluster provided error
        localXError = cluster->getClusterErrorX();
        MeasurementPoint mp(cluster->x(), cluster->y());  // Y error from topology
        float sigma_flat = 1.0f / std::sqrt(12.0f);
        float sigma2 = cluster->positionError(sigma_flat);
        sigma2 *= sigma2;
        MeasurementError posErr(sigma2, 0, sigma2);
        LocalError localErr = topo.localError(mp, posErr);
        localYError = std::sqrt(localErr.yy());
      }

      weightedErrorX2 +=
          energy * energy * localXError * localXError;  //approx -- lets say they're in the same ref frame
      weightedErrorY2 += energy * energy * localYError * localYError;
    }

    avgXError = std::sqrt(weightedErrorX2) / totalEnergy;
    avgYError = std::sqrt(weightedErrorY2) / totalEnergy;
  }

  DetId mergedId = DetId(seedId.rawId());
  FTLMergedCluster mergedCluster(
      mergedId, totalEnergy, avgTime, avgTimeError, avgX, avgY, avgXError, avgYError, clusterIds, clusterRefs);

  return mergedCluster;
}

void MTDMergedClusterProducer::produce(edm::Event& e, const edm::EventSetup& es) {
  // Get topology for navigation
  auto const& geom = es.getData(mtdgeoToken_);
  auto topologyHandle = es.getTransientHandle(mtdtopoToken_);
  const MTDTopology* topology = topologyHandle.product();

  edm::Handle<FTLClusterCollection> btlClustersHandle;
  edm::Handle<FTLClusterCollection> etlClustersHandle;
  e.getByToken(btlClustersToken_, btlClustersHandle);
  e.getByToken(etlClustersToken_, etlClustersHandle);

  LogTrace("MTDMergedClusterProducer") << "Processing event " << e.id();

  auto btlOutput = std::make_unique<FTLMergedClusterCollection>();
  auto etlOutput = std::make_unique<FTLMergedClusterCollection>();

  std::map<uint32_t, std::vector<FTLMergedCluster>> mergedByDet;
  std::map<uint32_t, std::vector<FTLMergedCluster>> etlByDet;

  if (!btlClustersHandle.isValid() || btlClustersHandle->empty()) {
    LogTrace("MTDMergedClusterProducer") << "No valid BTL clusters found in event " << e.id();
    e.put(std::move(btlOutput), btlMergedClusterInstance_);
  } else {
    // collect clusters and sort by module ID
    std::vector<const FTLCluster*> allClusters;
    std::map<BTLDetId, std::vector<const FTLCluster*>> clusterMap;

    for (const auto& detSet : *btlClustersHandle) {
      for (const auto& cluster : detSet) {
        if (cluster.energy() < energyThreshold_)
          continue;
        allClusters.push_back(&cluster);
        clusterMap[cluster.id()].push_back(&cluster);
      }
    }

    // sorting by rod and module

    std::sort(allClusters.begin(), allClusters.end(), [&topology](const FTLCluster* a, const FTLCluster* b) {
      BTLDetId idA(a->id());
      BTLDetId idB(b->id());
      auto [iphiA, ietaA] = topology->btlIndex(idA.geographicalId(BTLDetId::CrysLayout::v4).rawId());
      auto [iphiB, ietaB] = topology->btlIndex(idB.geographicalId(BTLDetId::CrysLayout::v4).rawId());

      if (iphiA != iphiB) {
        return iphiA < iphiB;
      }
      if (ietaA != ietaB) {
        return ietaA < ietaB;
      }
      return a->id().rawId() < b->id().rawId();
    });

    std::set<const FTLCluster*> processedClusters;

    // Process clusters
    for (const auto* cluster : allClusters) {
      if (processedClusters.count(cluster)) {
        continue;
      }

      BTLDetId cluId = cluster->id();
      std::vector<const FTLCluster*> mergedClusterClusters = {cluster};
      processedClusters.insert(cluster);

      // Get topology indices
      std::pair<uint32_t, uint32_t> indices = topology->btlIndex(cluId.rawId());
      uint32_t iphi = indices.first;
      uint32_t ieta = indices.second;

      if ((ieta == 48) ||
          (ieta ==
           96)) {  // Don't merge clusters in eta=0 due to gap in detectors, merging would be unphysical, just keep FTLClusters as they are. Also don't merge at the end of the trays, there is nothing to merge with
        FTLMergedCluster mergedCluster = mergeClusters(mergedClusterClusters, cluId, geom, btlClustersHandle);
        mergedByDet[mergedCluster.id().rawId()].push_back(std::move(mergedCluster));
        continue;
      }

      // Check for edge hits in current cluster
      bool edgeHitIn0 = false;
      bool edgeHitIn15 = false;

      for (int i = 0; i < cluster->size(); ++i) {
        auto hit = cluster->hit(i);
        int hit_col = hit.y();
        if (hit_col == 0) {
          edgeHitIn0 = true;
        } else if (hit_col == 15) {
          edgeHitIn15 = true;
        }
      }

      bool hasEdgeHitCurrent = false;
      if ((((ieta < 48) && edgeHitIn0)) || ((ieta > 48) && edgeHitIn15)) {
        hasEdgeHitCurrent =
            true;  // only consider edge hit in 0 direction for clusters with ieta<48, to avoid merging across the gap between modules
      }

      // ETA DIRECTION MERGING
      if (hasEdgeHitCurrent && iphi != std::numeric_limits<uint32_t>::max() &&
          ieta != std::numeric_limits<uint32_t>::max()) {
        //std::vector<int> etaOffsets = {1, -1};
        int etaOffset = 1;
        uint32_t adjDetIdRaw = topology->btlidFromIndex(iphi, ieta + etaOffset);
        if (adjDetIdRaw != 0) {
          BTLDetId adjDetId(adjDetIdRaw);
          auto it = clusterMap.find(adjDetId);
          if (it != clusterMap.end()) {
            for (const auto* adjClusterPtr : it->second) {
              if (processedClusters.count(adjClusterPtr)) {
                continue;
              }

              const FTLCluster* adjCluster = adjClusterPtr;

              // Check for opposite edge hit
              bool hasOppositeEdgeHit = false;
              for (int j = 0; j < adjCluster->size(); ++j) {
                auto hit = adjCluster->hit(j);
                int hit_col = hit.y();
                if ((edgeHitIn0 && hit_col == 15 && ieta < 48) || (edgeHitIn15 && hit_col == 0 && ieta > 48)) {
                  hasOppositeEdgeHit = true;
                  break;
                }
              }

              bool timeOk = areTimingCompatible(cluster, adjCluster);
              if (hasOppositeEdgeHit && timeOk) {
                mergedClusterClusters.push_back(adjCluster);
                // mark neighbor as processed immediately to avoid reuse
                processedClusters.insert(adjCluster);
              }
            }
          }
        }
      }

      // create mergedcluster from merged + leftover singles
      FTLMergedCluster mergedCluster = mergeClusters(mergedClusterClusters, cluId, geom, btlClustersHandle);
      mergedByDet[mergedCluster.id().rawId()].push_back(std::move(mergedCluster));
    }
    for (auto const& entry : mergedByDet) {
      uint32_t rawDetId = entry.first;
      const auto& vec = entry.second;
      edmNew::DetSetVector<FTLMergedCluster>::FastFiller filler(*btlOutput, rawDetId);
      for (const auto& mc : vec)
        filler.push_back(mc);
    }

    LogTrace("MTDMergedClusterProducer") << "About to put " << btlOutput->size() << " MergedClusters into event "
                                         << e.id() << std::endl;
    e.put(std::move(btlOutput), btlMergedClusterInstance_);
    LogTrace("MTDMergedClusterProducer") << "=== Successfully put BTL MergedClusters into event ===" << std::endl;
  }  // end of BTL processing

  // ETL processing - for now just convert to merged cluster format without actual merging:
  if (!etlClustersHandle.isValid() || etlClustersHandle->empty()) {
    LogTrace("MTDMergedClusterProducer") << "No valid ETL clusters found in event " << e.id() << std::endl;
    e.put(std::move(etlOutput), etlMergedClusterInstance_);
  } else {
    LogTrace("MTDMergedClusterProducer") << "Processing " << etlClustersHandle->size() << " ETL clusters in event "
                                         << e.id() << std::endl;
    for (const auto& detSet : *etlClustersHandle) {
      for (const auto& cluster : detSet) {
        if (cluster.energy() < energyThreshold_)
          continue;
        std::vector<const FTLCluster*> singleClusterVec = {&cluster};
        FTLMergedCluster mergedCluster = mergeClusters(singleClusterVec, cluster.id(), geom, etlClustersHandle);
        etlByDet[mergedCluster.id().rawId()].push_back(std::move(mergedCluster));
      }
    }

    for (auto const& entry : etlByDet) {
      uint32_t rawDetId = entry.first;
      const auto& vec = entry.second;
      edmNew::DetSetVector<FTLMergedCluster>::FastFiller filler(*etlOutput, rawDetId);
      for (const auto& mc : vec)
        filler.push_back(mc);
    }
    LogTrace("MTDMergedClusterProducer") << "About to put " << etlOutput->size() << " ETL MergedClusters into event "
                                         << e.id() << std::endl;
    e.put(std::move(etlOutput), etlMergedClusterInstance_);
    LogTrace("MTDMergedClusterProducer") << "=== Successfully put ETL MergedClusters into event ===" << std::endl;
  }
}

void MTDMergedClusterProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("btlBarrel", edm::InputTag("mtdClusters", "FTLBarrel"));
  desc.add<edm::InputTag>("etlEndcap", edm::InputTag("mtdClusters", "FTLEndcap"));
  desc.add<std::string>("btlMergedClusterInstance", "FTLBarrel");
  desc.add<std::string>("etlMergedClusterInstance", "FTLEndcap");
  desc.add<double>("timeThreshold", 10.0);
  desc.add<double>("energyThreshold", 0.0);  // MeV
  descriptions.add("MTDMergedClusterProducer", desc);
}

DEFINE_FWK_MODULE(MTDMergedClusterProducer);
