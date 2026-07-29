#define EDM_ML_DEBUG

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
#include "Geometry/CommonTopologies/interface/GeomDet.h"
#include "DataFormats/GeometryVector/interface/LocalPoint.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "DataFormats/GeometryCommonDetAlgo/interface/MeasurementPoint.h"
#include "Geometry/MTDGeometryBuilder/interface/RectangularMTDTopology.h"
#include "Geometry/MTDGeometryBuilder/interface/ProxyMTDTopology.h"
#include "Geometry/MTDCommonData/interface/MTDTopologyMode.h"


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
                                 const MTDGeometry& geom,
                                 edm::Handle<FTLClusterCollection> btlClustersHandle);
  std::vector<FTLClusterRef> clusterRefs;
  std::vector<DetId> hitDetId;
  std::vector<int> hitRow;
  std::vector<int> hitCol;
  std::vector<float> hitEnergy;
  std::vector<float> hitTime;
  std::vector<float> hitTimeError;

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
                                                         const MTDGeometry& geom,
                                                         edm::Handle<FTLClusterCollection> mtdClustersHandle) {
  float totalEnergy = 0;
  float weightedTime = 0;
  float weightedTimeError2 = 0;

  double weightedGlobalX = 0;
  double weightedGlobalY = 0;
  double weightedGlobalZ = 0;

  clusterRefs.clear();
  hitDetId.clear();
  hitRow.clear();
  hitCol.clear();
  hitEnergy.clear();
  hitTime.clear();
  hitTimeError.clear();

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
  DetId mergedId = DetId(primary->id().rawId());

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

    clusterRefs.push_back(edmNew::makeRefTo(mtdClustersHandle, cluster));

    for (int i = 0; i < cluster->size(); ++i) {
      hitDetId.emplace_back(cluster->id());
      auto thisHit = cluster->hit(i);
      hitRow.emplace_back(static_cast<int>(thisHit.x()));
      hitCol.emplace_back(static_cast<int>(thisHit.y()));
      hitEnergy.emplace_back(thisHit.energy());
      hitTime.emplace_back(thisHit.time());
      hitTimeError.emplace_back(thisHit.time_error());
    }

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
  const GeomDet* seedDet = geom.idToDetUnit(mergedId);
  if (seedDet && totalEnergy > 0) {
    LocalPoint lp = seedDet->surface().toLocal(avgGlobal);
    avgX = lp.x();  // along crystal (phi)
    avgY = lp.y();  // perpendicular to crystal (eta)
  } else {
    edm::LogWarning("MTDMergedClusterProducer") << "Unable to convert avgGlobal to seed-local coordinates for seedId "
                                                << mergedId.rawId() << " , geometry not available";
  }

  // errors:
  float avgXError = 0.f;
  float avgYError = 0.f;

  if (totalEnergy > 0) {
    double weightedErrorX2 = 0.;
    double weightedErrorY2 = 0.;

    for (const auto* cluster : clusters) {
#ifdef EDM_ML_DEBUG
      LogDebug("MTDMergedClusterProducer") << "Original hits for cluster in DetId: " << cluster->id();
      for (int ihit = 0; ihit < cluster->size(); ++ihit) {
        auto thisHit = cluster->hit(ihit);
        LogTrace("MTDMergedClusterProducer")
            << "Cluster hit " << ihit << " row/col = " << thisHit.x() << " " << thisHit.y()
            << " energy = " << thisHit.energy() << " time = " << thisHit.time() << " +/- " << thisHit.time_error();
        int hit_row = cluster->minHitRow() + cluster->hitOffset()[ihit * 2];
        int hit_col = cluster->minHitCol() + cluster->hitOffset()[ihit * 2 + 1];
        if (hit_row != thisHit.x() || hit_col != thisHit.y()) {
          edm::LogWarning("MTDMergedClusterProducer")
              << "Index in cluster memory not consistent, row/col = " << hit_row << " " << hit_col;
        }
      }
#endif
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

  FTLMergedCluster mergedCluster(
      mergedId, totalEnergy, avgTime, avgTimeError, avgX, avgY, avgXError, avgYError);

  for (size_t i = 0; i < hitDetId.size(); i++) {
    mergedCluster.addHit(hitDetId[i], hitRow[i], hitCol[i], hitTime[i], hitTimeError[i], hitEnergy[i]);
  }

  LogTrace("MTDMergedClusterProducer") << "Building merged cluster " << mergedCluster;
  return mergedCluster;
}

void MTDMergedClusterProducer::produce(edm::Event& e, const edm::EventSetup& es) {
  // Get topology for navigation
  auto const& geom = es.getData(mtdgeoToken_);
  auto topologyHandle = es.getTransientHandle(mtdtopoToken_);
  const MTDTopology* topology = topologyHandle.product();

  static constexpr uint32_t halfTrayBTL_SMidx = MTDTopology::BTLLayout::nBTLeta_ / 2;
  
  edm::Handle<FTLClusterCollection> btlClustersHandle;
  edm::Handle<FTLClusterCollection> etlClustersHandle;
  e.getByToken(btlClustersToken_, btlClustersHandle);
  e.getByToken(etlClustersToken_, etlClustersHandle);

  LogTrace("MTDMergedClusterProducer") << "Processing event " << e.id();

  auto btlOutput = std::make_unique<FTLMergedClusterCollection>();
  auto etlOutput = std::make_unique<FTLMergedClusterCollection>();

  std::map<uint32_t, std::vector<FTLMergedCluster>> mergedByDet;

  if (!btlClustersHandle.isValid() || btlClustersHandle->empty()) {
    LogTrace("MTDMergedClusterProducer") << "No valid BTL clusters found in event " << e.id();
    e.put(std::move(btlOutput), btlMergedClusterInstance_);
  } else {
    LogTrace("MTDMergedClusterProducer") << "Processing " << btlClustersHandle->size()
                                         << " BTL cluster DetSets in event " << e.id() << std::endl;
    // collect clusters and sort by module ID
    size_t totalClusters = 0;
    for (const auto& detSet : *btlClustersHandle) {
      totalClusters += detSet.size();
    }
    std::vector<const FTLCluster*> allClusters;
    allClusters.reserve(totalClusters);


    for (const auto& detSet : *btlClustersHandle) {
      for (const auto& cluster : detSet) {
        if (cluster.energy() < energyThreshold_)
          continue;
        allClusters.push_back(&cluster);
      }
    }
     
    // sorting by rod and module

    std::sort(allClusters.begin(), allClusters.end(), [&topology](const FTLCluster* a, const FTLCluster* b) {
      BTLDetId idA(a->id());
      BTLDetId idB(b->id());
      auto [iphiA, ietaA] = topology->btlIndex(idA.geographicalId(MTDTopologyMode::crysLayoutFromTopoMode(topology->getMTDTopologyMode())).rawId());
      auto [iphiB, ietaB] = topology->btlIndex(idB.geographicalId(MTDTopologyMode::crysLayoutFromTopoMode(topology->getMTDTopologyMode())).rawId());

      if (iphiA != iphiB) { 
        return iphiA < iphiB;
      }
      if (ietaA != ietaB) {
        return ietaA < ietaB;
      }
      if (ietaA < halfTrayBTL_SMidx){
        return a->minHitCol() > b->minHitCol();
      }else{
        return a->minHitCol() < b->minHitCol();
      }
    });

    
    bool alreadyMergedThisCluster = false;
    uint32_t currentRawId = 0;
    std::unique_ptr<edmNew::DetSetVector<FTLMergedCluster>::FastFiller> filler;
    size_t index(0);
    
    for (size_t i = 0; i < allClusters.size(); ++i) {
      const FTLCluster* cluster = allClusters[i];
      if (alreadyMergedThisCluster){
        alreadyMergedThisCluster = false; 
        continue;
      }
      
      BTLDetId cluId = cluster->id();
      std::vector<const FTLCluster*> mergedClusterClusters = {cluster};

      // Get topology indices
      std::pair<uint32_t, uint32_t> indices = topology->btlIndex(cluId.rawId());
      //uint32_t iphi = indices.first;
      uint32_t ieta = indices.second;
      uint32_t iphi = indices.first;
      
      bool hasEdgeHitCurrent = false;
      if (ieta < halfTrayBTL_SMidx){
        if (cluster->minHitCol() == 0){
          hasEdgeHitCurrent = true;
        } 
      }else if (ieta > halfTrayBTL_SMidx){
        if (cluster->maxHitCol() == 15){
          hasEdgeHitCurrent = true;
        }
      }
      // in the case ieta == halfTrayBTL_SMidx, merging would be unphysical due to gap in detectors, so we don't want to merge. In that case hasEdgeHitCurrent will be false.


      if (hasEdgeHitCurrent){
        bool hasEdgeHitNext = false;
        if (i + 1 < allClusters.size()) {
          const FTLCluster* nextCluster = allClusters[i + 1];
          BTLDetId nextDetId = nextCluster->id();
          std::pair<uint32_t, uint32_t> next_indices = topology->btlIndex(nextDetId.rawId());
          uint32_t ietanext = next_indices.second;
          uint32_t iphinext = next_indices.first;
          if (ieta == (ietanext - 1) && iphi == iphinext){
            if (ieta < halfTrayBTL_SMidx){
              if (nextCluster->maxHitCol() == 15){
                hasEdgeHitNext = true;
              }
            }else{
              if (nextCluster->minHitCol() == 0){
                hasEdgeHitNext = true;
              }
            }
          }
          if (hasEdgeHitNext){
            bool timeOk = areTimingCompatible(cluster, nextCluster);
            if (timeOk) {
              mergedClusterClusters.push_back(nextCluster);
              alreadyMergedThisCluster = true;
            }
        }
      }
      }

      // create mergedcluster from merged (or from single cluster if no merge happened)
      FTLMergedCluster mergedCluster = mergeClusters(mergedClusterClusters, geom, btlClustersHandle);
      uint32_t clusterId = mergedCluster.id().rawId();

      if (clusterId != currentRawId) {
        //new filler when changning the rawId
        filler.reset();
        filler = std::make_unique<edmNew::DetSetVector<FTLMergedCluster>::FastFiller>(*btlOutput, clusterId);
        currentRawId = clusterId;
      }
      LogDebug("MTDMergedClusterProducer") << "BTL merged cluster # " << std::setw(5) << index << " " << mergedCluster;
      filler->push_back(std::move(mergedCluster));
      index++;
    }
    filler.reset();

    LogTrace("MTDMergedClusterProducer") << "About to put " << btlOutput->size()
                                         << " BTL MergedCluster DetSets into event " << e.id() << std::endl;
    e.put(std::move(btlOutput), btlMergedClusterInstance_);
    LogTrace("MTDMergedClusterProducer") << "=== Successfully put BTL MergedClusters into event ===" << std::endl;
  }  // end of BTL processing

  // ETL processing - for now just convert to merged cluster format without actual merging:
  if (!etlClustersHandle.isValid() || etlClustersHandle->empty()) {
    LogTrace("MTDMergedClusterProducer") << "No valid ETL clusters found in event " << e.id() << std::endl;
    e.put(std::move(etlOutput), etlMergedClusterInstance_);
  } else {
    LogTrace("MTDMergedClusterProducer") << "Processing " << etlClustersHandle->size()
                                         << " ETL cluster DetSets in event " << e.id() << std::endl;
    std::vector<const FTLCluster*> singleClusterVec(1);
    size_t index(0);
    for (const auto& detSet : *etlClustersHandle) {
      edmNew::DetSetVector<FTLMergedCluster>::FastFiller filler(*etlOutput, detSet.id());
      for (const auto& cluster : detSet) {
        if (cluster.energy() < energyThreshold_)
          continue;
        singleClusterVec[0] = &cluster;
        FTLMergedCluster mergedCluster = mergeClusters(singleClusterVec, geom, etlClustersHandle);
        filler.push_back(mergedCluster);
        LogDebug("MTDMergedClusterProducer")
            << "ETL merged cluster # " << std::setw(5) << index << " " << mergedCluster;
        index++;
      }
    }

    LogTrace("MTDMergedClusterProducer") << "About to put " << etlOutput->size()
                                         << " ETL MergedCluster DetSets into event " << e.id() << std::endl;
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
