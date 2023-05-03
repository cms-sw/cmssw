#ifndef RecoLocalCalo_HGCalRecProducers_HGCalImagingAlgo_h
#define RecoLocalCalo_HGCalRecProducers_HGCalImagingAlgo_h

#include "RecoLocalCalo/HGCalRecProducers/interface/HGCalClusteringAlgoBase.h"

#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"

#include "Geometry/HGCalGeometry/interface/HGCalGeometry.h"
#include "Geometry/CaloTopology/interface/HGCalTopology.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/HGCRecHit/interface/HGCRecHitCollections.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloCellGeometry.h"
#include "Geometry/CaloGeometry/interface/TruncatedPyramid.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"

#include "DataFormats/Math/interface/Point3D.h"
#include "DataFormats/EgammaReco/interface/BasicCluster.h"

#include "RecoLocalCalo/HGCalRecAlgos/interface/RecHitTools.h"
#include "CommonTools/RecoAlgos/interface/KDTreeLinkerAlgo.h"

// C/C++ headers
#include <string>
#include <vector>
#include <set>

using Density = hgcal_clustering::Density;

class HGCalImagingAlgo : public HGCalClusteringAlgoBase {
public:
  HGCalImagingAlgo(const edm::ParameterSet &ps)
      : HGCalClusteringAlgoBase(
            (HGCalClusteringAlgoBase::VerbosityLevel)ps.getUntrackedParameter<unsigned int>("verbosity", 3),
            reco::CaloCluster::undefined),
        thresholdW0_(ps.getParameter<std::vector<double>>("thresholdW0")),
        positionDeltaRho_c_(ps.getParameter<std::vector<double>>("positionDeltaRho_c")),
        vecDeltas_(ps.getParameter<std::vector<double>>("deltac")),
        kappa_(ps.getParameter<double>("kappa")),
        ecut_(ps.getParameter<double>("ecut")),
        sigma2_(1.0),
        dependSensor_(ps.getParameter<bool>("dependSensor")),
        dEdXweights_(ps.getParameter<std::vector<double>>("dEdXweights")),
        thicknessCorrection_(ps.getParameter<std::vector<double>>("thicknessCorrection")),
        fcPerMip_(ps.getParameter<std::vector<double>>("fcPerMip")),
        fcPerEle_(ps.getParameter<double>("fcPerEle")),
        nonAgedNoises_(ps.getParameter<edm::ParameterSet>("noises").getParameter<std::vector<double>>("values")),
        noiseMip_(ps.getParameter<edm::ParameterSet>("noiseMip").getParameter<double>("noise_MIP")),
        initialized_(false) {}

  ~HGCalImagingAlgo() override {}

  void getEventSetupPerAlgorithm(const edm::EventSetup &es) override;

  void populate(const HGCRecHitCollection &hits) override;
  // this is the method that will start the clusterisation (it is possible to invoke this method more than once - but make sure it is with
  // different hit collections (or else use reset)

  void makeClusters() override;

  // this is the method to get the cluster collection out
  std::vector<reco::BasicCluster> getClusters(bool) override;

  // use this if you want to reuse the same cluster object but don't want to accumulate clusters (hardly useful?)
  void reset() override {
    clusters_v_.clear();
    clusters_v_.shrink_to_fit();
    layerClustersPerLayer_.clear();
    layerClustersPerLayer_.shrink_to_fit();
    for (auto &it : points_) {
      it.clear();
      it.shrink_to_fit();
      std::vector<KDNode>().swap(it);
    }
    for (unsigned int i = 0; i < minpos_.size(); i++) {
      minpos_[i][0] = 0.;
      minpos_[i][1] = 0.;
      maxpos_[i][0] = 0.;
      maxpos_[i][1] = 0.;
    }
  }
  void computeThreshold();

  //getDensity
  Density getDensity() override;

  static void fillPSetDescription(edm::ParameterSetDescription &iDesc) {
    iDesc.add<std::vector<double>>("thresholdW0", {2.9, 2.9, 2.9});
    iDesc.add<std::vector<double>>("positionDeltaRho_c", {1.3, 1.3, 1.3});
    iDesc.add<std::vector<double>>("deltac",
                                   {
                                       2.0,
                                       2.0,
                                       5.0,
                                   });
    iDesc.add<bool>("dependSensor", true);
    iDesc.add<double>("ecut", 3.0);
    iDesc.add<double>("kappa", 9.0);
    iDesc.addUntracked<unsigned int>("verbosity", 3);
    iDesc.add<std::vector<double>>("dEdXweights", {});
    iDesc.add<std::vector<double>>("thicknessCorrection", {});
    iDesc.add<std::vector<double>>("fcPerMip", {});
    iDesc.add<double>("fcPerEle", 0.0);
    edm::ParameterSetDescription descNestedNoises;
    descNestedNoises.add<std::vector<double>>("values", {});
    iDesc.add<edm::ParameterSetDescription>("noises", descNestedNoises);
    edm::ParameterSetDescription descNestedNoiseMIP;
    descNestedNoiseMIP.add<bool>("scaleByDose", false);
    descNestedNoiseMIP.add<double>("scaleByDoseFactor", 1.);
    iDesc.add<edm::ParameterSetDescription>("scaleByDose", descNestedNoiseMIP);
    descNestedNoiseMIP.add<std::string>("doseMap", "");
    iDesc.add<edm::ParameterSetDescription>("doseMap", descNestedNoiseMIP);
    descNestedNoiseMIP.add<double>("noise_MIP", 1. / 100.);
    iDesc.add<edm::ParameterSetDescription>("noiseMip", descNestedNoiseMIP);
  }

  /// point in the space
  typedef math::XYZPoint Point;

private:
  // To compute the cluster position
  std::vector<double> thresholdW0_;
  std::vector<double> positionDeltaRho_c_;

  // The two parameters used to identify clusters
  std::vector<double> vecDeltas_;
  double kappa_;

  // The hit energy cutoff
  double ecut_;

  // for energy sharing
  double sigma2_;  // transverse shower size

  // The vector of clusters
  std::vector<reco::BasicCluster> clusters_v_;

  // For keeping the density per hit
  Density density_;

  // various parameters used for calculating the noise levels for a given sensor (and whether to use them)
  bool dependSensor_;
  std::vector<double> dEdXweights_;
  std::vector<double> thicknessCorrection_;
  std::vector<double> fcPerMip_;
  double fcPerEle_;
  std::vector<double> nonAgedNoises_;
  double noiseMip_;
  std::vector<std::vector<double>> thresholds_;
  std::vector<std::vector<double>> sigmaNoise_;

  // initialization bool
  bool initialized_;

  struct Hexel {
    double x;
    double y;
    double z;
    bool isHalfCell;
    double weight;
    double fraction;
    DetId detid;
    double rho;
    double delta;
    int nearestHigher;
    bool isBorder;
    bool isHalo;
    int clusterIndex;
    float sigmaNoise;
    float thickness;
    const hgcal::RecHitTools *tools;

    Hexel(const HGCRecHit &hit,
          DetId id_in,
          bool isHalf,
          float sigmaNoise_in,
          float thickness_in,
          const hgcal::RecHitTools *tools_in)
        : isHalfCell(isHalf),
          weight(0.),
          fraction(1.0),
          detid(id_in),
          rho(0.),
          delta(0.),
          nearestHigher(-1),
          isBorder(false),
          isHalo(false),
          clusterIndex(-1),
          sigmaNoise(sigmaNoise_in),
          thickness(thickness_in),
          tools(tools_in) {
      const GlobalPoint position(tools->getPosition(detid));
      weight = hit.energy();
      x = position.x();
      y = position.y();
      z = position.z();
    }
    Hexel()
        : x(0.),
          y(0.),
          z(0.),
          isHalfCell(false),
          weight(0.),
          fraction(1.0),
          detid(),
          rho(0.),
          delta(0.),
          nearestHigher(-1),
          isBorder(false),
          isHalo(false),
          clusterIndex(-1),
          sigmaNoise(0.),
          thickness(0.),
          tools(nullptr) {}
    bool operator>(const Hexel &rhs) const { return (rho > rhs.rho); }
  };

  typedef KDTreeLinkerAlgo<Hexel> KDTree;
  typedef KDTreeNodeInfo<Hexel> KDNode;

  std::vector<std::vector<std::vector<KDNode>>> layerClustersPerLayer_;

  std::vector<size_t> sort_by_delta(const std::vector<KDNode> &v) const {
    std::vector<size_t> idx(v.size());
    std::iota(std::begin(idx), std::end(idx), 0);
    sort(idx.begin(), idx.end(), [&v](size_t i1, size_t i2) { return v[i1].data.delta > v[i2].data.delta; });
    return idx;
  }

  std::vector<std::vector<KDNode>> points_;  //a vector of vectors of hexels, one for each layer
  //@@EM todo: the number of layers should be obtained programmatically - the range is 1-n instead of 0-n-1...

  std::vector<std::array<float, 2>> minpos_;
  std::vector<std::array<float, 2>> maxpos_;

  //these functions should be in a helper class.
  inline double distance2(const Hexel &pt1, const Hexel &pt2) const {  //distance squared
    const double dx = pt1.x - pt2.x;
    const double dy = pt1.y - pt2.y;
    return (dx * dx + dy * dy);
  }                                                                   //distance squaredq
  inline double distance(const Hexel &pt1, const Hexel &pt2) const {  //2-d distance on the layer (x-y)
    return std::sqrt(distance2(pt1, pt2));
  }
  double calculateLocalDensity(std::vector<KDNode> &, KDTree &, const unsigned int) const;  //return max density
  double calculateDistanceToHigher(std::vector<KDNode> &) const;
  int findAndAssignClusters(std::vector<KDNode> &,
                            KDTree &,
                            double,
                            KDTreeBox<2> &,
                            const unsigned int,
                            std::vector<std::vector<KDNode>> &) const;
  math::XYZPoint calculatePosition(std::vector<KDNode> &) const;

  //For keeping the density information
  void setDensity(const std::vector<KDNode> &nd);

  // attempt to find subclusters within a given set of hexels
  std::vector<unsigned> findLocalMaximaInCluster(const std::vector<KDNode> &);
  math::XYZPoint calculatePositionWithFraction(const std::vector<KDNode> &, const std::vector<double> &);
  double calculateEnergyWithFraction(const std::vector<KDNode> &, const std::vector<double> &);
  // outputs
  void shareEnergy(const std::vector<KDNode> &, const std::vector<unsigned> &, std::vector<std::vector<double>> &);
};

#endif
