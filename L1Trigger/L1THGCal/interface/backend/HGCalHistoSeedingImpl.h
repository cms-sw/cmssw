#ifndef __L1Trigger_L1THGCal_HGCalHistoSeedingImpl_h__
#define __L1Trigger_L1THGCal_HGCalHistoSeedingImpl_h__

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/L1THGCal/interface/HGCalCluster.h"
#include "DataFormats/L1THGCal/interface/HGCalMulticluster.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "L1Trigger/L1THGCal/interface/HGCalTriggerGeometryBase.h"
#include "L1Trigger/L1THGCal/interface/backend/HGCalShowerShape.h"
#include "L1Trigger/L1THGCal/interface/HGCalTriggerTools.h"
#include "L1Trigger/L1THGCal/interface/backend/HGCalTriggerClusterIdentificationBase.h"

class HGCalHistoSeedingImpl {
public:
  HGCalHistoSeedingImpl(const edm::ParameterSet& conf);

  void eventSetup(const edm::EventSetup& es) { triggerTools_.eventSetup(es); }

  float dR(const l1t::HGCalCluster& clu, const GlobalPoint& seed) const;

  void findHistoSeeds(const std::vector<edm::Ptr<l1t::HGCalCluster>>& clustersPtr,
                      std::vector<std::pair<GlobalPoint, double>>& seedPositionsEnergy);

private:
  enum SeedingType { HistoMaxC3d, HistoSecondaryMaxC3d, HistoThresholdC3d, HistoInterpolatedMaxC3d };

  typedef std::map<std::array<int, 3>, float> Histogram;

  Histogram fillHistoClusters(const std::vector<edm::Ptr<l1t::HGCalCluster>>& clustersPtrs);

  Histogram fillSmoothPhiHistoClusters(const Histogram& histoClusters, const vector<unsigned>& binSums);

  Histogram fillSmoothRPhiHistoClusters(const Histogram& histoClusters);

  std::vector<std::pair<GlobalPoint, double>> computeMaxSeeds(const Histogram& histoClusters);

  std::vector<std::pair<GlobalPoint, double>> computeSecondaryMaxSeeds(const Histogram& histoClusters);

  std::vector<std::pair<GlobalPoint, double>> computeInterpolatedMaxSeeds(const Histogram& histoClusters);

  std::vector<std::pair<GlobalPoint, double>> computeThresholdSeeds(const Histogram& histoClusters);

  std::string seedingAlgoType_;
  SeedingType seedingType_;

  unsigned nBinsRHisto_ = 36;
  unsigned nBinsPhiHisto_ = 216;
  std::vector<unsigned> binsSumsHisto_;
  double histoThreshold_ = 20.;
  std::vector<double> neighbour_weights_;

  HGCalTriggerTools triggerTools_;

  static constexpr unsigned neighbour_weights_size_ = 9;
  static constexpr double kROverZMin_ = 0.09;
  static constexpr double kROverZMax_ = 0.52;
};

#endif
