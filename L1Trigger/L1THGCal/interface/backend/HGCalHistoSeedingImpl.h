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
private:
  struct Bin {
    float sumMipPt = 0.;
    float weighted_x = 0.;
    float weighted_y = 0.;
  };
  template <typename T>
  class HistogramT {
  public:
    using Data = std::vector<T>;
    using iterator = typename Data::iterator;
    using const_iterator = typename Data::const_iterator;

  public:
    HistogramT(unsigned bins1, unsigned bins2)
        : bins1_(bins1), bins2_(bins2), bins_(bins1 * bins2), histogram_(bins_ * kSides_) {}

    T& at(int zside, unsigned x1, unsigned x2) { return histogram_[index(zside, x1, x2)]; }

    const T& at(int zside, unsigned x1, unsigned x2) const { return histogram_.at(index(zside, x1, x2)); }

    iterator begin() { return histogram_.begin(); }
    const_iterator begin() const { return histogram_.begin(); }
    iterator end() { return histogram_.end(); }
    const_iterator end() const { return histogram_.end(); }

  private:
    static constexpr unsigned kSides_ = 2;
    unsigned bins1_ = 0;
    unsigned bins2_ = 0;
    unsigned bins_ = 0;
    Data histogram_;

    unsigned index(int zside, unsigned x1, unsigned x2) const {
      if (x1 >= bins1_ || x2 >= bins2_) {
        throw cms::Exception("OutOfBound") << "Trying to access bin (" << x1 << "," << x2
                                           << ") in seeding histogram of size (" << bins1_ << "," << bins2_ << ")";
      }
      return x2 + bins2_ * x1 + bins_ * (zside > 0 ? 1 : 0);
    }
  };
  using Histogram = HistogramT<Bin>;

public:
  HGCalHistoSeedingImpl(const edm::ParameterSet& conf);

  void eventSetup(const edm::EventSetup& es) { triggerTools_.eventSetup(es); }

  float dR(const l1t::HGCalCluster& clu, const GlobalPoint& seed) const;

  void findHistoSeeds(const std::vector<edm::Ptr<l1t::HGCalCluster>>& clustersPtr,
                      std::vector<std::pair<GlobalPoint, double>>& seedPositionsEnergy);

private:
  enum SeedingType { HistoMaxC3d, HistoSecondaryMaxC3d, HistoThresholdC3d, HistoInterpolatedMaxC3d };
  enum SeedingPosition { BinCentre, TCWeighted };

  Histogram fillHistoClusters(const std::vector<edm::Ptr<l1t::HGCalCluster>>& clustersPtrs);

  Histogram fillSmoothPhiHistoClusters(const Histogram& histoClusters, const vector<unsigned>& binSums);

  Histogram fillSmoothRPhiHistoClusters(const Histogram& histoClusters);

  void setSeedEnergyAndPosition(std::vector<std::pair<GlobalPoint, double>>& seedPositionsEnergy,
                                int z_side,
                                unsigned bin_R,
                                unsigned bin_phi,
                                const Bin& histBin);

  std::vector<std::pair<GlobalPoint, double>> computeMaxSeeds(const Histogram& histoClusters);

  std::vector<std::pair<GlobalPoint, double>> computeSecondaryMaxSeeds(const Histogram& histoClusters);

  std::vector<std::pair<GlobalPoint, double>> computeInterpolatedMaxSeeds(const Histogram& histoClusters);

  std::vector<std::pair<GlobalPoint, double>> computeThresholdSeeds(const Histogram& histoClusters);

  std::string seedingAlgoType_;
  SeedingType seedingType_;
  SeedingPosition seedingPosition_;

  unsigned nBinsRHisto_ = 42;
  unsigned nBinsPhiHisto_ = 216;
  std::vector<unsigned> binsSumsHisto_;
  double histoThreshold_ = 20.;
  std::vector<double> neighbour_weights_;

  HGCalTriggerTools triggerTools_;

  static constexpr unsigned neighbour_weights_size_ = 9;
  static constexpr double kROverZMin_ = 0.076;
  static constexpr double kROverZMax_ = 0.58;
};

#endif
