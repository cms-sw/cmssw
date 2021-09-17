#include "L1Trigger/L1THGCal/interface/backend/HGCalClusteringDummyImpl.h"
#include <unordered_map>
#include <unordered_set>
#include "DataFormats/Common/interface/OrphanHandle.h"
#include "DataFormats/Common/interface/PtrVector.h"

// class constructor
HGCalClusteringDummyImpl::HGCalClusteringDummyImpl(const edm::ParameterSet& conf)
    : calibSF_(conf.getParameter<double>("calibSF_cluster")),
      layerWeights_(conf.getParameter<std::vector<double>>("layerWeights")),
      applyLayerWeights_(conf.getParameter<bool>("applyLayerCalibration")) {
  edm::LogInfo("HGCalClusterParameters") << "C2d global calibration factor: " << calibSF_;
}

// Create one cluster per TC for direct TC->3D clustering
void HGCalClusteringDummyImpl::clusterizeDummy(const std::vector<edm::Ptr<l1t::HGCalTriggerCell>>& triggerCellsPtrs,
                                               l1t::HGCalClusterBxCollection& clusters) {
  std::vector<l1t::HGCalCluster> clustersTmp;
  for (std::vector<edm::Ptr<l1t::HGCalTriggerCell>>::const_iterator tc = triggerCellsPtrs.begin();
       tc != triggerCellsPtrs.end();
       ++tc) {
    clustersTmp.emplace_back(*tc);
  }

  /* store clusters in the persistent collection */
  clusters.resize(0, clustersTmp.size());
  for (unsigned i(0); i < clustersTmp.size(); ++i) {
    calibratePt(clustersTmp.at(i));
    clusters.set(0, i, clustersTmp.at(i));
  }
}

void HGCalClusteringDummyImpl::calibratePt(l1t::HGCalCluster& cluster) {
  double calibPt = 0.;

  if (applyLayerWeights_ && !triggerTools_.isNose(cluster.detId())) {
    unsigned layerN = triggerTools_.layerWithOffset(cluster.detId());

    if (layerWeights_.at(layerN) == 0.) {
      throw cms::Exception("BadConfiguration")
          << "2D cluster energy forced to 0 by calibration coefficients.\n"
          << "The configuration should be changed. "
          << "Discarded layers should be defined in hgcalTriggerGeometryESProducer.TriggerGeometry.DisconnectedLayers "
             "and not with calibration coefficients = 0\n";
    }

    calibPt = layerWeights_.at(layerN) * cluster.mipPt();

  }

  else {
    calibPt = cluster.pt() * calibSF_;
  }

  cluster.setPt(calibPt);
}
