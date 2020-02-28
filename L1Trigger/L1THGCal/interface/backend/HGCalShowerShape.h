#ifndef __L1Trigger_L1THGCal_HGCalShowerShape_h__
#define __L1Trigger_L1THGCal_HGCalShowerShape_h__
#include <vector>
#include <functional>
#include <cmath>
#include "DataFormats/L1THGCal/interface/HGCalMulticluster.h"
#include "DataFormats/Math/interface/LorentzVector.h"
#include "L1Trigger/L1THGCal/interface/HGCalTriggerGeometryBase.h"
#include "L1Trigger/L1THGCal/interface/HGCalTriggerTools.h"
#include "DataFormats/Math/interface/deltaPhi.h"

class HGCalShowerShape {
public:
  typedef math::XYZTLorentzVector LorentzVector;

  HGCalShowerShape() : threshold_(0.) {}
  HGCalShowerShape(const edm::ParameterSet& conf);

  ~HGCalShowerShape() {}

  void eventSetup(const edm::EventSetup& es) { triggerTools_.eventSetup(es); }

  int firstLayer(const l1t::HGCalMulticluster& c3d) const;
  int lastLayer(const l1t::HGCalMulticluster& c3d) const;
  int maxLayer(const l1t::HGCalMulticluster& c3d) const;
  int showerLength(const l1t::HGCalMulticluster& c3d) const {
    return lastLayer(c3d) - firstLayer(c3d) + 1;
  }  //in number of layers
  // Maximum number of consecutive layers in the cluster
  int coreShowerLength(const l1t::HGCalMulticluster& c3d, const HGCalTriggerGeometryBase& triggerGeometry) const;
  float percentileLayer(const l1t::HGCalMulticluster& c3d,
                        const HGCalTriggerGeometryBase& triggerGeometry,
                        float quantile = 0.5) const;

  float percentileTriggerCells(const l1t::HGCalMulticluster& c3d, float quantile = 0.5) const;

  float eMax(const l1t::HGCalMulticluster& c3d) const;

  float meanZ(const l1t::HGCalMulticluster& c3d) const;
  float sigmaZZ(const l1t::HGCalMulticluster& c3d) const;

  float sigmaEtaEtaTot(const l1t::HGCalMulticluster& c3d) const;
  float sigmaEtaEtaTot(const l1t::HGCalCluster& c2d) const;
  float sigmaEtaEtaMax(const l1t::HGCalMulticluster& c3d) const;

  float sigmaPhiPhiTot(const l1t::HGCalMulticluster& c3d) const;
  float sigmaPhiPhiTot(const l1t::HGCalCluster& c2d) const;
  float sigmaPhiPhiMax(const l1t::HGCalMulticluster& c3d) const;

  float sigmaRRTot(const l1t::HGCalMulticluster& c3d) const;
  float sigmaRRTot(const l1t::HGCalCluster& c2d) const;
  float sigmaRRMax(const l1t::HGCalMulticluster& c3d) const;
  float sigmaRRMean(const l1t::HGCalMulticluster& c3d, float radius = 5.) const;

  void fillShapes(l1t::HGCalMulticluster&, const HGCalTriggerGeometryBase&) const;

private:
  float meanX(const std::vector<pair<float, float>>& energy_X_tc) const;
  // Compute energy-weighted RMS of any variable X in the cluster
  // Delta(a,b) functor as template argument. Default is (a-b)
  template <typename Delta = std::minus<float>>
  float sigmaXX(const std::vector<pair<float, float>>& energy_X_tc, const float X_cluster) const {
    Delta delta;
    float Etot = 0;
    float deltaX2_sum = 0;
    for (const auto& energy_X : energy_X_tc) {
      deltaX2_sum += energy_X.first * pow(delta(energy_X.second, X_cluster), 2);
      Etot += energy_X.first;
    }
    float X_MSE = 0;
    if (Etot > 0)
      X_MSE = deltaX2_sum / Etot;
    float X_RMS = sqrt(X_MSE);
    return X_RMS;
  }
  // Special case of delta for phi
  template <class T>
  struct DeltaPhi {
    T operator()(const T& x, const T& y) const { return deltaPhi(x, y); }
  };
  float sigmaPhiPhi(const std::vector<pair<float, float>>& energy_phi_tc, const float phi_cluster) const {
    return sigmaXX<DeltaPhi<float>>(energy_phi_tc, phi_cluster);
  }
  template <typename T, typename Tref>
  bool pass(const T& obj, const Tref& ref) const {
    bool pass_threshold = (obj.mipPt() > threshold_);
    GlobalPoint proj(Basic3DVector<float>(obj.position()) / std::abs(obj.position().z()));
    bool pass_distance = ((proj - ref.centreProj()).mag() < distance_);
    return pass_threshold && pass_distance;
  }

  HGCalTriggerTools triggerTools_;
  double threshold_ = 0.;
  double distance_ = 1.;
};

#endif
