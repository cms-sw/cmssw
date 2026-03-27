#ifndef RecoLocalCalo_HGCalRecProducers_HGCalClusteringAlgoBase_h
#define RecoLocalCalo_HGCalRecProducers_HGCalClusteringAlgoBase_h

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"

#include "DataFormats/HGCRecHit/interface/HGCRecHitCollections.h"
#include "DataFormats/ParticleFlowReco/interface/PFRecHitFwd.h"
#include "DataFormats/Math/interface/Point3D.h"
#include "DataFormats/EgammaReco/interface/BasicCluster.h"

#include "RecoLocalCalo/HGCalRecAlgos/interface/RecHitTools.h"

#include "CondFormats/DataRecord/interface/EcalPFRecHitThresholdsRcd.h"
#include "CondFormats/EcalObjects/interface/EcalPFRecHitThresholds.h"
#include "CondFormats/DataRecord/interface/HcalPFCutsRcd.h"
#include "CondFormats/HcalObjects/interface/HcalPFCuts.h"

// C/C++ headers
#include <vector>
#include <numeric>

namespace hgcal_clustering {
  template <typename T>
  std::vector<size_t> sorted_indices(const std::vector<T> &v) {
    // initialize original index locations
    std::vector<size_t> idx(v.size());
    std::iota(std::begin(idx), std::end(idx), 0);

    // sort indices based on comparing values in v
    std::sort(idx.begin(), idx.end(), [&v](size_t i1, size_t i2) { return v[i1] > v[i2]; });

    return idx;
  }

  template <typename T>
  size_t max_index(const std::vector<T> &v) {
    // initialize original index locations
    std::vector<size_t> idx(v.size(), 0);
    std::iota(std::begin(idx), std::end(idx), 0);

    // take the max index based on comparing values in v
    auto maxidx = std::max_element(
        idx.begin(), idx.end(), [&v](size_t i1, size_t i2) { return v[i1].data.rho < v[i2].data.rho; });

    return (*maxidx);
  }

  //Density collection
  typedef std::map<DetId, float> Density;

};  // namespace hgcal_clustering

class HGCalClusteringAlgoBase {
public:
  enum VerbosityLevel { pDEBUG = 0, pWARNING = 1, pINFO = 2, pERROR = 3 };

  HGCalClusteringAlgoBase(VerbosityLevel v, reco::CaloCluster::AlgoId algo) : verbosity_(v), algoId_(algo) {}
  virtual ~HGCalClusteringAlgoBase() {}

  virtual void populate(const HGCRecHitCollection &hits) = 0;
  virtual void populate(const reco::PFRecHitCollection &hits) = 0;
  virtual void makeClusters() = 0;
  virtual std::vector<reco::BasicCluster> getClusters(bool) = 0;
  virtual void reset() = 0;
  virtual hgcal_clustering::Density getDensity() { return {}; };        //implementation is in some child class
  virtual void getEventSetupPerAlgorithm(const edm::EventSetup &es) {}  //implementation is in some child class

  inline void getEventSetup(const edm::EventSetup &es, hgcal::RecHitTools rhtools) {
    rhtools_ = rhtools;
    maxlayer_ = rhtools_.lastLayer(isNose_);
    lastLayerEE_ = rhtools_.lastLayerEE(isNose_);
    lastLayerFH_ = rhtools_.lastLayerFH();
    firstLayerBH_ = rhtools_.firstLayerBH();
    scintMaxIphi_ = rhtools_.getScintMaxIphi();
    getEventSetupPerAlgorithm(es);
  }
  inline void setVerbosity(VerbosityLevel the_verbosity) { verbosity_ = the_verbosity; }
  inline void setAlgoId(reco::CaloCluster::AlgoId algo, bool isNose = false) {
    algoId_ = algo;
    isNose_ = isNose;
  }
  virtual void setThresholds(edm::ESGetToken<EcalPFRecHitThresholds, EcalPFRecHitThresholdsRcd>,
                             edm::ESGetToken<HcalPFCuts, HcalPFCutsRcd>){};

  //max number of layers
  unsigned int maxlayer_ = 0;
  // last layer per subdetector
  unsigned int lastLayerEE_ = 0;
  unsigned int lastLayerFH_ = 0;
  unsigned int firstLayerBH_ = 0;
  int scintMaxIphi_ = 0;
  bool isNose_ = false;

protected:
  // The verbosity level
  VerbosityLevel verbosity_;

  // The vector of clusters
  std::vector<reco::BasicCluster> clusters_v_;

  hgcal::RecHitTools rhtools_;

  // The algo id
  reco::CaloCluster::AlgoId algoId_;

  edm::ESGetToken<CaloGeometry, CaloGeometryRecord> caloGeomToken_;
};

#endif
