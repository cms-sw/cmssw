#include "L1Trigger/L1THGCal/interface/backend/HGCalHistoSeedingImpl.h"
#include "L1Trigger/L1THGCal/interface/backend/HGCalShowerShape.h"
#include "DataFormats/Math/interface/deltaR.h"
#include "DataFormats/Math/interface/deltaPhi.h"
#include <numeric>

HGCalHistoSeedingImpl::HGCalHistoSeedingImpl(const edm::ParameterSet& conf)
    : seedingAlgoType_(conf.getParameter<std::string>("type_histoalgo")),
      nBins1_(conf.getParameter<unsigned>("nBins_X1_histo_multicluster")),
      nBins2_(conf.getParameter<unsigned>("nBins_X2_histo_multicluster")),
      binsSumsHisto_(conf.getParameter<std::vector<unsigned>>("binSumsHisto")),
      histoThreshold_(conf.getParameter<double>("threshold_histo_multicluster")),
      neighbour_weights_(conf.getParameter<std::vector<double>>("neighbour_weights")),
      smoothing_ecal_(conf.getParameter<std::vector<double>>("seed_smoothing_ecal")),
      smoothing_hcal_(conf.getParameter<std::vector<double>>("seed_smoothing_hcal")),
      kROverZMin_(conf.getParameter<double>("kROverZMin")),
      kROverZMax_(conf.getParameter<double>("kROverZMax")) {
  if (seedingAlgoType_ == "HistoMaxC3d") {
    seedingType_ = HistoMaxC3d;
  } else if (seedingAlgoType_ == "HistoSecondaryMaxC3d") {
    seedingType_ = HistoSecondaryMaxC3d;
  } else if (seedingAlgoType_ == "HistoThresholdC3d") {
    seedingType_ = HistoThresholdC3d;
  } else if (seedingAlgoType_ == "HistoInterpolatedMaxC3d") {
    seedingType_ = HistoInterpolatedMaxC3d;
  } else {
    throw cms::Exception("HGCTriggerParameterError") << "Unknown Multiclustering type '" << seedingAlgoType_;
  }

  if (conf.getParameter<std::string>("seed_position") == "BinCentre") {
    seedingPosition_ = BinCentre;
  } else if (conf.getParameter<std::string>("seed_position") == "TCWeighted") {
    seedingPosition_ = TCWeighted;
  } else {
    throw cms::Exception("HGCTriggerParameterError")
        << "Unknown Seed Position option '" << conf.getParameter<std::string>("seed_position");
  }
  if (conf.getParameter<std::string>("seeding_space") == "RPhi") {
    seedingSpace_ = RPhi;
    navigator_ = Navigator(nBins1_, Navigator::AxisType::Bounded, nBins2_, Navigator::AxisType::Circular);
  } else if (conf.getParameter<std::string>("seeding_space") == "XY") {
    seedingSpace_ = XY;
    navigator_ = Navigator(nBins1_, Navigator::AxisType::Bounded, nBins2_, Navigator::AxisType::Bounded);
  } else {
    throw cms::Exception("HGCTriggerParameterError")
        << "Unknown seeding space  '" << conf.getParameter<std::string>("seeding_space");
  }

  edm::LogInfo("HGCalMulticlusterParameters")
      << "\nMulticluster number of X1-bins for the histo algorithm: " << nBins1_
      << "\nMulticluster number of X2-bins for the histo algorithm: " << nBins2_
      << "\nMulticluster MIPT threshold for histo threshold algorithm: " << histoThreshold_
      << "\nMulticluster type of multiclustering algortihm: " << seedingAlgoType_;

  if (seedingAlgoType_.find("Histo") != std::string::npos && seedingSpace_ == RPhi &&
      nBins1_ != binsSumsHisto_.size()) {
    throw cms::Exception("Inconsistent bin size")
        << "Inconsistent nBins_X1_histo_multicluster ( " << nBins1_ << " ) and binSumsHisto ( " << binsSumsHisto_.size()
        << " ) size in HGCalMulticlustering\n";
  }

  if (neighbour_weights_.size() != neighbour_weights_size_) {
    throw cms::Exception("Inconsistent vector size")
        << "Inconsistent size of neighbour weights vector in HGCalMulticlustering ( " << neighbour_weights_.size()
        << " ). Should be " << neighbour_weights_size_ << "\n";
  }
}

HGCalHistoSeedingImpl::Histogram HGCalHistoSeedingImpl::fillHistoClusters(
    const std::vector<edm::Ptr<l1t::HGCalCluster>>& clustersPtrs) {
  Histogram histoClusters(nBins1_, nBins2_);
  std::array<double, 4> bounds = boundaries();
  double minx1 = std::get<0>(bounds);
  double maxx1 = std::get<1>(bounds);
  double minx2 = std::get<2>(bounds);
  double maxx2 = std::get<3>(bounds);

  for (auto& clu : clustersPtrs) {
    float x1 = 0., x2 = 0;
    switch (seedingSpace_) {
      case RPhi:
        x1 = sqrt(pow(clu->centreProj().x(), 2) + pow(clu->centreProj().y(), 2));
        x2 = reco::reduceRange(clu->phi());
        break;
      case XY:
        x1 = clu->centreProj().x();
        x2 = clu->centreProj().y();
        break;
    };
    if (x1 < minx1 || x1 >= maxx1) {
      throw cms::Exception("OutOfBound") << "TC X1 = " << x1 << " out of the seeding histogram bounds " << minx1
                                         << " - " << maxx1;
    }
    if (x2 < minx2 || x2 >= maxx2) {
      throw cms::Exception("OutOfBound") << "TC X2 = " << x2 << " out of the seeding histogram bounds " << minx2
                                         << " - " << maxx2;
    }
    unsigned bin1 = unsigned((x1 - minx1) * nBins1_ / (maxx1 - minx1));
    unsigned bin2 = unsigned((x2 - minx2) * nBins2_ / (maxx2 - minx2));

    auto& bin = histoClusters.at(triggerTools_.zside(clu->detId()), bin1, bin2);
    bin.values[Bin::Content::Sum] += clu->mipPt();
    if (triggerTools_.isEm(clu->detId())) {
      bin.values[Bin::Content::Ecal] += clu->mipPt();
    } else {
      bin.values[Bin::Content::Hcal] += clu->mipPt();
    }
    bin.weighted_x += (clu->centreProj().x()) * clu->mipPt();
    bin.weighted_y += (clu->centreProj().y()) * clu->mipPt();
  }

  for (auto& bin : histoClusters) {
    bin.weighted_x /= bin.values[Bin::Content::Sum];
    bin.weighted_y /= bin.values[Bin::Content::Sum];
  }

  return histoClusters;
}

HGCalHistoSeedingImpl::Histogram HGCalHistoSeedingImpl::fillSmoothHistoClusters(const Histogram& histoClusters,
                                                                                const vector<double>& kernel,
                                                                                Bin::Content binContent) {
  Histogram histoSmooth(histoClusters);

  unsigned kernel_size = std::sqrt(kernel.size());
  if (kernel_size * kernel_size != kernel.size()) {
    throw cms::Exception("HGCTriggerParameterError") << "Only square kernels can be used.";
  }
  if (kernel_size % 2 != 1) {
    throw cms::Exception("HGCTriggerParameterError") << "The kernel size must be an odd value.";
  }
  int shift_max = (kernel_size - 1) / 2;
  double normalization = std::accumulate(kernel.begin(), kernel.end(), 0.);
  for (int z_side : {-1, 1}) {
    for (unsigned x1 = 0; x1 < nBins1_; x1++) {
      for (unsigned x2 = 0; x2 < nBins2_; x2++) {
        const auto& bin_orig = histoClusters.at(z_side, x1, x2);
        double smooth = 0.;
        navigator_.setHome(x1, x2);
        for (int x1_shift = -shift_max; x1_shift <= shift_max; x1_shift++) {
          int index1 = x1_shift + shift_max;
          for (int x2_shift = -shift_max; x2_shift <= shift_max; x2_shift++) {
            auto shifted = navigator_.move(x1_shift, x2_shift);
            int index2 = x2_shift + shift_max;
            double kernel_value = kernel.at(index1 * kernel_size + index2);
            bool out = shifted[0] == -1 || shifted[1] == -1;
            double content = (out ? 0. : histoClusters.at(z_side, shifted[0], shifted[1]).values[binContent]);
            smooth += content * kernel_value;
          }
        }
        auto& bin = histoSmooth.at(z_side, x1, x2);
        bin.values[binContent] = smooth / normalization;
        bin.weighted_x = bin_orig.weighted_x;
        bin.weighted_y = bin_orig.weighted_y;
      }
    }
  }

  return histoSmooth;
}

HGCalHistoSeedingImpl::Histogram HGCalHistoSeedingImpl::fillSmoothPhiHistoClusters(const Histogram& histoClusters,
                                                                                   const vector<unsigned>& binSums) {
  Histogram histoSumPhiClusters(nBins1_, nBins2_);

  for (int z_side : {-1, 1}) {
    for (unsigned bin1 = 0; bin1 < nBins1_; bin1++) {
      int nBinsSide = (binSums[bin1] - 1) / 2;
      float R1 = kROverZMin_ + bin1 * (kROverZMax_ - kROverZMin_);
      float R2 = R1 + (kROverZMax_ - kROverZMin_);
      double area =
          0.5 * (pow(R2, 2) - pow(R1, 2)) *
          (1 +
           0.5 *
               (1 -
                pow(0.5,
                    nBinsSide)));  // Takes into account different area of bins in different R-rings + sum of quadratic weights used

      for (unsigned bin2 = 0; bin2 < nBins2_; bin2++) {
        const auto& bin_orig = histoClusters.at(z_side, bin1, bin2);
        float content = bin_orig.values[Bin::Content::Sum];

        for (int bin22 = 1; bin22 <= nBinsSide; bin22++) {
          int binToSumLeft = bin2 - bin22;
          if (binToSumLeft < 0)
            binToSumLeft += nBins2_;
          unsigned binToSumRight = bin2 + bin22;
          if (binToSumRight >= nBins2_)
            binToSumRight -= nBins2_;

          content += histoClusters.at(z_side, bin1, binToSumLeft).values[Bin::Content::Sum] /
                     pow(2, bin22);  // quadratic kernel

          content += histoClusters.at(z_side, bin1, binToSumRight).values[Bin::Content::Sum] /
                     pow(2, bin22);  // quadratic kernel
        }

        auto& bin = histoSumPhiClusters.at(z_side, bin1, bin2);
        bin.values[Bin::Content::Sum] = content / area;
        bin.weighted_x = bin_orig.weighted_x;
        bin.weighted_y = bin_orig.weighted_y;
      }
    }
  }

  return histoSumPhiClusters;
}

HGCalHistoSeedingImpl::Histogram HGCalHistoSeedingImpl::fillSmoothRPhiHistoClusters(const Histogram& histoClusters) {
  Histogram histoSumRPhiClusters(nBins1_, nBins2_);

  for (int z_side : {-1, 1}) {
    for (unsigned bin1 = 0; bin1 < nBins1_; bin1++) {
      float weight =
          (bin1 == 0 || bin1 == nBins1_ - 1) ? 1.5 : 2.;  //Take into account edges with only one side up or down

      for (unsigned bin2 = 0; bin2 < nBins2_; bin2++) {
        const auto& bin_orig = histoClusters.at(z_side, bin1, bin2);
        float content = bin_orig.values[Bin::Content::Sum];
        float contentDown = bin1 > 0 ? histoClusters.at(z_side, bin1 - 1, bin2).values[Bin::Content::Sum] : 0;
        float contentUp = bin1 < (nBins1_ - 1) ? histoClusters.at(z_side, bin1 + 1, bin2).values[Bin::Content::Sum] : 0;

        auto& bin = histoSumRPhiClusters.at(z_side, bin1, bin2);
        bin.values[Bin::Content::Sum] = (content + 0.5 * contentDown + 0.5 * contentUp) / weight;
        bin.weighted_x = bin_orig.weighted_x;
        bin.weighted_y = bin_orig.weighted_y;
      }
    }
  }

  return histoSumRPhiClusters;
}

void HGCalHistoSeedingImpl::setSeedEnergyAndPosition(std::vector<std::pair<GlobalPoint, double>>& seedPositionsEnergy,
                                                     int z_side,
                                                     unsigned bin1,
                                                     unsigned bin2,
                                                     const Bin& histBin) {
  float x_seed = 0;
  float y_seed = 0;
  std::array<double, 4> bounds = boundaries();
  double minx1 = std::get<0>(bounds);
  double maxx1 = std::get<1>(bounds);
  double minx2 = std::get<2>(bounds);
  double maxx2 = std::get<3>(bounds);

  if (seedingPosition_ == BinCentre) {
    float x1_seed = minx1 + (bin1 + 0.5) * (maxx1 - minx1) / nBins1_;
    float x2_seed = minx2 + (bin2 + 0.5) * (maxx2 - minx2) / nBins2_;
    switch (seedingSpace_) {
      case RPhi:
        x_seed = x1_seed * cos(x2_seed);
        y_seed = x1_seed * sin(x2_seed);
        break;
      case XY:
        x_seed = x1_seed;
        y_seed = x2_seed;
    };
  } else if (seedingPosition_ == TCWeighted) {
    x_seed = histBin.weighted_x;
    y_seed = histBin.weighted_y;
  }

  seedPositionsEnergy.emplace_back(GlobalPoint(x_seed, y_seed, z_side), histBin.values[Bin::Content::Sum]);
}

std::vector<std::pair<GlobalPoint, double>> HGCalHistoSeedingImpl::computeMaxSeeds(const Histogram& histoClusters) {
  std::vector<std::pair<GlobalPoint, double>> seedPositionsEnergy;

  for (int z_side : {-1, 1}) {
    for (unsigned bin1 = 0; bin1 < nBins1_; bin1++) {
      for (unsigned bin2 = 0; bin2 < nBins2_; bin2++) {
        float MIPT_seed = histoClusters.at(z_side, bin1, bin2).values[Bin::Content::Sum];
        bool isMax = MIPT_seed > histoThreshold_;
        if (!isMax)
          continue;

        navigator_.setHome(bin1, bin2);
        auto pos_N = navigator_.move(1, 0);
        auto pos_S = navigator_.move(-1, 0);
        auto pos_W = navigator_.move(0, -1);
        auto pos_E = navigator_.move(0, 1);
        auto pos_NW = navigator_.move(1, -1);
        auto pos_NE = navigator_.move(1, 1);
        auto pos_SW = navigator_.move(-1, -1);
        auto pos_SE = navigator_.move(-1, 1);

        float MIPT_N = (pos_N[0] != -1 && pos_N[1] != -1)
                           ? histoClusters.at(z_side, pos_N[0], pos_N[1]).values[Bin::Content::Sum]
                           : 0;
        float MIPT_S = (pos_S[0] != -1 && pos_S[1] != -1)
                           ? histoClusters.at(z_side, pos_S[0], pos_S[1]).values[Bin::Content::Sum]
                           : 0;
        float MIPT_W = (pos_W[0] != -1 && pos_W[1] != -1)
                           ? histoClusters.at(z_side, pos_W[0], pos_W[1]).values[Bin::Content::Sum]
                           : 0;
        float MIPT_E = (pos_E[0] != -1 && pos_E[1] != -1)
                           ? histoClusters.at(z_side, pos_E[0], pos_E[1]).values[Bin::Content::Sum]
                           : 0;
        float MIPT_NW = (pos_NW[0] != -1 && pos_NW[1] != -1)
                            ? histoClusters.at(z_side, pos_NW[0], pos_NW[1]).values[Bin::Content::Sum]
                            : 0;
        float MIPT_NE = (pos_NE[0] != -1 && pos_NE[1] != -1)
                            ? histoClusters.at(z_side, pos_NE[0], pos_NE[1]).values[Bin::Content::Sum]
                            : 0;
        float MIPT_SW = (pos_SW[0] != -1 && pos_SW[1] != -1)
                            ? histoClusters.at(z_side, pos_SW[0], pos_SW[1]).values[Bin::Content::Sum]
                            : 0;
        float MIPT_SE = (pos_SE[0] != -1 && pos_SE[1] != -1)
                            ? histoClusters.at(z_side, pos_SE[0], pos_SE[1]).values[Bin::Content::Sum]
                            : 0;

        isMax &= MIPT_seed >= MIPT_S && MIPT_seed > MIPT_N && MIPT_seed >= MIPT_E && MIPT_seed >= MIPT_SE &&
                 MIPT_seed >= MIPT_NE && MIPT_seed > MIPT_W && MIPT_seed > MIPT_SW && MIPT_seed > MIPT_NW;

        if (isMax) {
          setSeedEnergyAndPosition(seedPositionsEnergy, z_side, bin1, bin2, histoClusters.at(z_side, bin1, bin2));
        }
      }
    }
  }

  return seedPositionsEnergy;
}

std::vector<std::pair<GlobalPoint, double>> HGCalHistoSeedingImpl::computeInterpolatedMaxSeeds(
    const Histogram& histoClusters) {
  std::vector<std::pair<GlobalPoint, double>> seedPositionsEnergy;

  for (int z_side : {-1, 1}) {
    for (unsigned bin1 = 0; bin1 < nBins1_; bin1++) {
      for (unsigned bin2 = 0; bin2 < nBins2_; bin2++) {
        float MIPT_seed = histoClusters.at(z_side, bin1, bin2).values[Bin::Content::Sum];

        navigator_.setHome(bin1, bin2);
        auto pos_N = navigator_.move(1, 0);
        auto pos_S = navigator_.move(-1, 0);
        auto pos_W = navigator_.move(0, -1);
        auto pos_E = navigator_.move(0, 1);
        auto pos_NW = navigator_.move(1, -1);
        auto pos_NE = navigator_.move(1, 1);
        auto pos_SW = navigator_.move(-1, -1);
        auto pos_SE = navigator_.move(-1, 1);

        float MIPT_N = (pos_N[0] != -1 && pos_N[1] != -1)
                           ? histoClusters.at(z_side, pos_N[0], pos_N[1]).values[Bin::Content::Sum]
                           : 0;
        float MIPT_S = (pos_S[0] != -1 && pos_S[1] != -1)
                           ? histoClusters.at(z_side, pos_S[0], pos_S[1]).values[Bin::Content::Sum]
                           : 0;
        float MIPT_W = (pos_W[0] != -1 && pos_W[1] != -1)
                           ? histoClusters.at(z_side, pos_W[0], pos_W[1]).values[Bin::Content::Sum]
                           : 0;
        float MIPT_E = (pos_E[0] != -1 && pos_E[1] != -1)
                           ? histoClusters.at(z_side, pos_E[0], pos_E[1]).values[Bin::Content::Sum]
                           : 0;
        float MIPT_NW = (pos_NW[0] != -1 && pos_NW[1] != -1)
                            ? histoClusters.at(z_side, pos_NW[0], pos_NW[1]).values[Bin::Content::Sum]
                            : 0;
        float MIPT_NE = (pos_NE[0] != -1 && pos_NE[1] != -1)
                            ? histoClusters.at(z_side, pos_NE[0], pos_NE[1]).values[Bin::Content::Sum]
                            : 0;
        float MIPT_SW = (pos_SW[0] != -1 && pos_SW[1] != -1)
                            ? histoClusters.at(z_side, pos_SW[0], pos_SW[1]).values[Bin::Content::Sum]
                            : 0;
        float MIPT_SE = (pos_SE[0] != -1 && pos_SE[1] != -1)
                            ? histoClusters.at(z_side, pos_SE[0], pos_SE[1]).values[Bin::Content::Sum]
                            : 0;

        float MIPT_pred = neighbour_weights_.at(0) * MIPT_NW + neighbour_weights_.at(1) * MIPT_N +
                          neighbour_weights_.at(2) * MIPT_NE + neighbour_weights_.at(3) * MIPT_W +
                          neighbour_weights_.at(5) * MIPT_E + neighbour_weights_.at(6) * MIPT_SW +
                          neighbour_weights_.at(7) * MIPT_S + neighbour_weights_.at(8) * MIPT_SE;

        bool isMax = MIPT_seed >= (MIPT_pred + histoThreshold_);

        if (isMax) {
          setSeedEnergyAndPosition(seedPositionsEnergy, z_side, bin1, bin2, histoClusters.at(z_side, bin1, bin2));
        }
      }
    }
  }

  return seedPositionsEnergy;
}

std::vector<std::pair<GlobalPoint, double>> HGCalHistoSeedingImpl::computeThresholdSeeds(
    const Histogram& histoClusters) {
  std::vector<std::pair<GlobalPoint, double>> seedPositionsEnergy;

  for (int z_side : {-1, 1}) {
    for (unsigned bin1 = 0; bin1 < nBins1_; bin1++) {
      for (unsigned bin2 = 0; bin2 < nBins2_; bin2++) {
        float MIPT_seed = histoClusters.at(z_side, bin1, bin2).values[Bin::Content::Sum];
        bool isSeed = MIPT_seed > histoThreshold_;

        if (isSeed) {
          setSeedEnergyAndPosition(seedPositionsEnergy, z_side, bin1, bin2, histoClusters.at(z_side, bin1, bin2));
        }
      }
    }
  }

  return seedPositionsEnergy;
}

std::vector<std::pair<GlobalPoint, double>> HGCalHistoSeedingImpl::computeSecondaryMaxSeeds(
    const Histogram& histoClusters) {
  std::vector<std::pair<GlobalPoint, double>> seedPositionsEnergy;

  HistogramT<uint8_t> primarySeedPositions(nBins1_, nBins2_);
  HistogramT<uint8_t> vetoPositions(nBins1_, nBins2_);

  //Search for primary seeds
  for (int z_side : {-1, 1}) {
    for (unsigned bin1 = 0; bin1 < nBins1_; bin1++) {
      for (unsigned bin2 = 0; bin2 < nBins2_; bin2++) {
        float MIPT_seed = histoClusters.at(z_side, bin1, bin2).values[Bin::Content::Sum];
        bool isMax = MIPT_seed > histoThreshold_;

        if (!isMax)
          continue;

        float MIPT_S = bin1 < (nBins1_ - 1) ? histoClusters.at(z_side, bin1 + 1, bin2).values[Bin::Content::Sum] : 0;
        float MIPT_N = bin1 > 0 ? histoClusters.at(z_side, bin1 - 1, bin2).values[Bin::Content::Sum] : 0;

        int binLeft = bin2 - 1;
        if (binLeft < 0)
          binLeft += nBins2_;
        unsigned binRight = bin2 + 1;
        if (binRight >= nBins2_)
          binRight -= nBins2_;

        float MIPT_W = histoClusters.at(z_side, bin1, binLeft).values[Bin::Content::Sum];
        float MIPT_E = histoClusters.at(z_side, bin1, binRight).values[Bin::Content::Sum];
        float MIPT_NW = bin1 > 0 ? histoClusters.at(z_side, bin1 - 1, binLeft).values[Bin::Content::Sum] : 0;
        float MIPT_NE = bin1 > 0 ? histoClusters.at(z_side, bin1 - 1, binRight).values[Bin::Content::Sum] : 0;
        float MIPT_SW =
            bin1 < (nBins1_ - 1) ? histoClusters.at(z_side, bin1 + 1, binLeft).values[Bin::Content::Sum] : 0;
        float MIPT_SE =
            bin1 < (nBins1_ - 1) ? histoClusters.at(z_side, bin1 + 1, binRight).values[Bin::Content::Sum] : 0;

        isMax &= MIPT_seed >= MIPT_S && MIPT_seed > MIPT_N && MIPT_seed >= MIPT_E && MIPT_seed >= MIPT_SE &&
                 MIPT_seed >= MIPT_NE && MIPT_seed > MIPT_W && MIPT_seed > MIPT_SW && MIPT_seed > MIPT_NW;

        if (isMax) {
          setSeedEnergyAndPosition(seedPositionsEnergy, z_side, bin1, bin2, histoClusters.at(z_side, bin1, bin2));

          primarySeedPositions.at(z_side, bin1, bin2) = true;

          vetoPositions.at(z_side, bin1, binLeft) = true;
          vetoPositions.at(z_side, bin1, binRight) = true;
          if (bin1 > 0) {
            vetoPositions.at(z_side, bin1 - 1, bin2) = true;
            vetoPositions.at(z_side, bin1 - 1, binRight) = true;
            vetoPositions.at(z_side, bin1 - 1, binLeft) = true;
          }
          if (bin1 < (nBins1_ - 1)) {
            vetoPositions.at(z_side, bin1 + 1, bin2) = true;
            vetoPositions.at(z_side, bin1 + 1, binRight) = true;
            vetoPositions.at(z_side, bin1 + 1, binLeft) = true;
          }
        }
      }
    }
  }

  //Search for secondary seeds

  for (int z_side : {-1, 1}) {
    for (unsigned bin1 = 0; bin1 < nBins1_; bin1++) {
      for (unsigned bin2 = 0; bin2 < nBins2_; bin2++) {
        //Cannot be a secondary seed if already a primary seed, or adjacent to primary seed
        if (primarySeedPositions.at(z_side, bin1, bin2) || vetoPositions.at(z_side, bin1, bin2))
          continue;

        float MIPT_seed = histoClusters.at(z_side, bin1, bin2).values[Bin::Content::Sum];
        bool isMax = MIPT_seed > histoThreshold_;

        float MIPT_S = bin1 < (nBins1_ - 1) ? histoClusters.at(z_side, bin1 + 1, bin2).values[Bin::Content::Sum] : 0;
        float MIPT_N = bin1 > 0 ? histoClusters.at(z_side, bin1 - 1, bin2).values[Bin::Content::Sum] : 0;

        int binLeft = bin2 - 1;
        if (binLeft < 0)
          binLeft += nBins2_;
        unsigned binRight = bin2 + 1;
        if (binRight >= nBins2_)
          binRight -= nBins2_;

        float MIPT_W = histoClusters.at(z_side, bin1, binLeft).values[Bin::Content::Sum];
        float MIPT_E = histoClusters.at(z_side, bin1, binRight).values[Bin::Content::Sum];
        float MIPT_NW = bin1 > 0 ? histoClusters.at(z_side, bin1 - 1, binLeft).values[Bin::Content::Sum] : 0;
        float MIPT_NE = bin1 > 0 ? histoClusters.at(z_side, bin1 - 1, binRight).values[Bin::Content::Sum] : 0;
        float MIPT_SW =
            bin1 < (nBins1_ - 1) ? histoClusters.at(z_side, bin1 + 1, binLeft).values[Bin::Content::Sum] : 0;
        float MIPT_SE =
            bin1 < (nBins1_ - 1) ? histoClusters.at(z_side, bin1 + 1, binRight).values[Bin::Content::Sum] : 0;

        isMax &= (((bin1 < nBins1_ - 1) && vetoPositions.at(z_side, bin1 + 1, bin2)) or MIPT_seed >= MIPT_S) &&
                 (((bin1 > 0) && vetoPositions.at(z_side, bin1 - 1, bin2)) or MIPT_seed > MIPT_N) &&
                 ((vetoPositions.at(z_side, bin1, binRight)) or MIPT_seed >= MIPT_E) &&
                 (((bin1 < nBins1_ - 1) && vetoPositions.at(z_side, bin1 + 1, binRight)) or MIPT_seed >= MIPT_SE) &&
                 (((bin1 > 0) && vetoPositions.at(z_side, bin1 - 1, binRight)) or MIPT_seed >= MIPT_NE) &&
                 ((vetoPositions.at(z_side, bin1, binLeft)) or MIPT_seed > MIPT_W) &&
                 (((bin1 < nBins1_ - 1) && vetoPositions.at(z_side, bin1 + 1, binLeft)) or MIPT_seed > MIPT_SW) &&
                 (((bin1 > 0) && vetoPositions.at(z_side, bin1 - 1, binLeft)) or MIPT_seed > MIPT_NW);

        if (isMax) {
          setSeedEnergyAndPosition(seedPositionsEnergy, z_side, bin1, bin2, histoClusters.at(z_side, bin1, bin2));
        }
      }
    }
  }

  return seedPositionsEnergy;
}

void HGCalHistoSeedingImpl::findHistoSeeds(const std::vector<edm::Ptr<l1t::HGCalCluster>>& clustersPtrs,
                                           std::vector<std::pair<GlobalPoint, double>>& seedPositionsEnergy) {
  /* put clusters into an r/z x phi histogram */
  Histogram histoCluster = fillHistoClusters(clustersPtrs);

  Histogram smoothHistoCluster;
  if (seedingSpace_ == RPhi) {
    /* smoothen along the phi direction + normalize each bin to same area */
    Histogram smoothPhiHistoCluster = fillSmoothPhiHistoClusters(histoCluster, binsSumsHisto_);

    /* smoothen along the r/z direction */
    smoothHistoCluster = fillSmoothRPhiHistoClusters(smoothPhiHistoCluster);
  } else if (seedingSpace_ == XY) {
    smoothHistoCluster = fillSmoothHistoClusters(histoCluster, smoothing_ecal_, Bin::Content::Ecal);
    smoothHistoCluster = fillSmoothHistoClusters(smoothHistoCluster, smoothing_hcal_, Bin::Content::Hcal);
    // Update sum with smoothen ECAL + HCAL
    for (int z_side : {-1, 1}) {
      for (unsigned x1 = 0; x1 < nBins1_; x1++) {
        for (unsigned x2 = 0; x2 < nBins2_; x2++) {
          auto& bin = smoothHistoCluster.at(z_side, x1, x2);
          bin.values[Bin::Content::Sum] = bin.values[Bin::Content::Ecal] + bin.values[Bin::Content::Hcal];
        }
      }
    }
  }

  /* seeds determined with local maximum criteria */
  if (seedingType_ == HistoMaxC3d)
    seedPositionsEnergy = computeMaxSeeds(smoothHistoCluster);
  else if (seedingType_ == HistoThresholdC3d)
    seedPositionsEnergy = computeThresholdSeeds(smoothHistoCluster);
  else if (seedingType_ == HistoInterpolatedMaxC3d)
    seedPositionsEnergy = computeInterpolatedMaxSeeds(smoothHistoCluster);
  else if (seedingType_ == HistoSecondaryMaxC3d)
    seedPositionsEnergy = computeSecondaryMaxSeeds(smoothHistoCluster);
}

std::array<double, 4> HGCalHistoSeedingImpl::boundaries() {
  switch (seedingSpace_) {
    case RPhi:
      return {{kROverZMin_, kROverZMax_, -M_PI, M_PI}};
    case XY:
      return {{-kXYMax_, kXYMax_, -kXYMax_, kXYMax_}};
  }
  return {{0., 0., 0., 0.}};
}
