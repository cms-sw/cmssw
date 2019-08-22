#include "L1Trigger/L1THGCal/interface/backend/HGCalHistoSeedingImpl.h"
#include "L1Trigger/L1THGCal/interface/backend/HGCalShowerShape.h"
#include "DataFormats/Math/interface/deltaR.h"
#include "DataFormats/Math/interface/deltaPhi.h"

HGCalHistoSeedingImpl::HGCalHistoSeedingImpl(const edm::ParameterSet& conf)
    : seedingAlgoType_(conf.getParameter<string>("type_multicluster")),
      nBinsRHisto_(conf.getParameter<unsigned>("nBins_R_histo_multicluster")),
      nBinsPhiHisto_(conf.getParameter<unsigned>("nBins_Phi_histo_multicluster")),
      binsSumsHisto_(conf.getParameter<std::vector<unsigned>>("binSumsHisto")),
      histoThreshold_(conf.getParameter<double>("threshold_histo_multicluster")),
      neighbour_weights_(conf.getParameter<std::vector<double>>("neighbour_weights")) {
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

  if (conf.getParameter<string>("seed_position") == "BinCentre") {
    seedingPosition_ = BinCentre;
  } else if (conf.getParameter<string>("seed_position") == "TCWeighted") {
    seedingPosition_ = TCWeighted;
  } else {
    throw cms::Exception("HGCTriggerParameterError")
        << "Unknown Seed Position option '" << conf.getParameter<string>("seed_position");
  }

  edm::LogInfo("HGCalMulticlusterParameters")
      << "\nMulticluster number of R-bins for the histo algorithm: " << nBinsRHisto_
      << "\nMulticluster number of Phi-bins for the histo algorithm: " << nBinsPhiHisto_
      << "\nMulticluster MIPT threshold for histo threshold algorithm: " << histoThreshold_
      << "\nMulticluster type of multiclustering algortihm: " << seedingAlgoType_;

  if (seedingAlgoType_.find("Histo") != std::string::npos && nBinsRHisto_ != binsSumsHisto_.size()) {
    throw cms::Exception("Inconsistent bin size")
        << "Inconsistent nBins_R_histo_multicluster ( " << nBinsRHisto_ << " ) and binSumsHisto ( "
        << binsSumsHisto_.size() << " ) size in HGCalMulticlustering\n";
  }

  if (neighbour_weights_.size() != neighbour_weights_size_) {
    throw cms::Exception("Inconsistent vector size")
        << "Inconsistent size of neighbour weights vector in HGCalMulticlustering ( " << neighbour_weights_.size()
        << " ). Should be " << neighbour_weights_size_ << "\n";
  }
}

HGCalHistoSeedingImpl::Histogram HGCalHistoSeedingImpl::fillHistoClusters(
    const std::vector<edm::Ptr<l1t::HGCalCluster>>& clustersPtrs) {
  Histogram histoClusters(nBinsRHisto_, nBinsPhiHisto_);

  for (auto& clu : clustersPtrs) {
    float ROverZ = sqrt(pow(clu->centreProj().x(), 2) + pow(clu->centreProj().y(), 2));
    if (ROverZ < kROverZMin_ || ROverZ >= kROverZMax_) {
      throw cms::Exception("OutOfBound") << "TC R/Z = " << ROverZ << " out of the seeding histogram bounds "
                                         << kROverZMin_ << " - " << kROverZMax_;
    }
    unsigned bin_R = unsigned((ROverZ - kROverZMin_) * nBinsRHisto_ / (kROverZMax_ - kROverZMin_));
    unsigned bin_phi = unsigned((reco::reduceRange(clu->phi()) + M_PI) * nBinsPhiHisto_ / (2 * M_PI));

    auto& bin = histoClusters.at(triggerTools_.zside(clu->detId()), bin_R, bin_phi);
    bin.sumMipPt += clu->mipPt();
    bin.weighted_x += (clu->centreProj().x()) * clu->mipPt();
    bin.weighted_y += (clu->centreProj().y()) * clu->mipPt();
  }

  for (auto& bin : histoClusters) {
    bin.weighted_x /= bin.sumMipPt;
    bin.weighted_y /= bin.sumMipPt;
  }

  return histoClusters;
}

HGCalHistoSeedingImpl::Histogram HGCalHistoSeedingImpl::fillSmoothPhiHistoClusters(const Histogram& histoClusters,
                                                                                   const vector<unsigned>& binSums) {
  Histogram histoSumPhiClusters(nBinsRHisto_, nBinsPhiHisto_);

  for (int z_side : {-1, 1}) {
    for (unsigned bin_R = 0; bin_R < nBinsRHisto_; bin_R++) {
      int nBinsSide = (binSums[bin_R] - 1) / 2;
      float R1 = kROverZMin_ + bin_R * (kROverZMax_ - kROverZMin_);
      float R2 = R1 + (kROverZMax_ - kROverZMin_);
      double area =
          0.5 * (pow(R2, 2) - pow(R1, 2)) *
          (1 +
           0.5 *
               (1 -
                pow(0.5,
                    nBinsSide)));  // Takes into account different area of bins in different R-rings + sum of quadratic weights used

      for (unsigned bin_phi = 0; bin_phi < nBinsPhiHisto_; bin_phi++) {
        const auto& bin_orig = histoClusters.at(z_side, bin_R, bin_phi);
        float content = bin_orig.sumMipPt;

        for (int bin_phi2 = 1; bin_phi2 <= nBinsSide; bin_phi2++) {
          int binToSumLeft = bin_phi - bin_phi2;
          if (binToSumLeft < 0)
            binToSumLeft += nBinsPhiHisto_;
          unsigned binToSumRight = bin_phi + bin_phi2;
          if (binToSumRight >= nBinsPhiHisto_)
            binToSumRight -= nBinsPhiHisto_;

          content += histoClusters.at(z_side, bin_R, binToSumLeft).sumMipPt / pow(2, bin_phi2);  // quadratic kernel

          content += histoClusters.at(z_side, bin_R, binToSumRight).sumMipPt / pow(2, bin_phi2);  // quadratic kernel
        }

        auto& bin = histoSumPhiClusters.at(z_side, bin_R, bin_phi);
        bin.sumMipPt = content / area;
        bin.weighted_x = bin_orig.weighted_x;
        bin.weighted_y = bin_orig.weighted_y;
      }
    }
  }

  return histoSumPhiClusters;
}

HGCalHistoSeedingImpl::Histogram HGCalHistoSeedingImpl::fillSmoothRPhiHistoClusters(const Histogram& histoClusters) {
  Histogram histoSumRPhiClusters(nBinsRHisto_, nBinsPhiHisto_);

  for (int z_side : {-1, 1}) {
    for (unsigned bin_R = 0; bin_R < nBinsRHisto_; bin_R++) {
      float weight =
          (bin_R == 0 || bin_R == nBinsRHisto_ - 1) ? 1.5 : 2.;  //Take into account edges with only one side up or down

      for (unsigned bin_phi = 0; bin_phi < nBinsPhiHisto_; bin_phi++) {
        const auto& bin_orig = histoClusters.at(z_side, bin_R, bin_phi);
        float content = bin_orig.sumMipPt;
        float contentDown = bin_R > 0 ? histoClusters.at(z_side, bin_R - 1, bin_phi).sumMipPt : 0;
        float contentUp = bin_R < (nBinsRHisto_ - 1) ? histoClusters.at(z_side, bin_R + 1, bin_phi).sumMipPt : 0;

        auto& bin = histoSumRPhiClusters.at(z_side, bin_R, bin_phi);
        bin.sumMipPt = (content + 0.5 * contentDown + 0.5 * contentUp) / weight;
        bin.weighted_x = bin_orig.weighted_x;
        bin.weighted_y = bin_orig.weighted_y;
      }
    }
  }

  return histoSumRPhiClusters;
}

void HGCalHistoSeedingImpl::setSeedEnergyAndPosition(std::vector<std::pair<GlobalPoint, double>>& seedPositionsEnergy,
                                                     int z_side,
                                                     unsigned bin_R,
                                                     unsigned bin_phi,
                                                     const Bin& histBin) {
  float x_seed = 0;
  float y_seed = 0;

  if (seedingPosition_ == BinCentre) {
    float ROverZ_seed = kROverZMin_ + (bin_R + 0.5) * (kROverZMax_ - kROverZMin_) / nBinsRHisto_;
    float phi_seed = -M_PI + (bin_phi + 0.5) * 2 * M_PI / nBinsPhiHisto_;
    x_seed = ROverZ_seed * cos(phi_seed);
    y_seed = ROverZ_seed * sin(phi_seed);
  } else if (seedingPosition_ == TCWeighted) {
    x_seed = histBin.weighted_x;
    y_seed = histBin.weighted_y;
  }

  seedPositionsEnergy.emplace_back(GlobalPoint(x_seed, y_seed, z_side), histBin.sumMipPt);
}

std::vector<std::pair<GlobalPoint, double>> HGCalHistoSeedingImpl::computeMaxSeeds(const Histogram& histoClusters) {
  std::vector<std::pair<GlobalPoint, double>> seedPositionsEnergy;

  for (int z_side : {-1, 1}) {
    for (unsigned bin_R = 0; bin_R < nBinsRHisto_; bin_R++) {
      for (unsigned bin_phi = 0; bin_phi < nBinsPhiHisto_; bin_phi++) {
        float MIPT_seed = histoClusters.at(z_side, bin_R, bin_phi).sumMipPt;
        bool isMax = MIPT_seed > histoThreshold_;
        if (!isMax)
          continue;

        float MIPT_S = bin_R < (nBinsRHisto_ - 1) ? histoClusters.at(z_side, bin_R + 1, bin_phi).sumMipPt : 0;
        float MIPT_N = bin_R > 0 ? histoClusters.at(z_side, bin_R - 1, bin_phi).sumMipPt : 0;

        int binLeft = bin_phi - 1;
        if (binLeft < 0)
          binLeft += nBinsPhiHisto_;
        unsigned binRight = bin_phi + 1;
        if (binRight >= nBinsPhiHisto_)
          binRight -= nBinsPhiHisto_;

        float MIPT_W = histoClusters.at(z_side, bin_R, binLeft).sumMipPt;
        float MIPT_E = histoClusters.at(z_side, bin_R, binRight).sumMipPt;
        float MIPT_NW = bin_R > 0 ? histoClusters.at(z_side, bin_R - 1, binLeft).sumMipPt : 0;
        float MIPT_NE = bin_R > 0 ? histoClusters.at(z_side, bin_R - 1, binRight).sumMipPt : 0;
        float MIPT_SW = bin_R < (nBinsRHisto_ - 1) ? histoClusters.at(z_side, bin_R + 1, binLeft).sumMipPt : 0;
        float MIPT_SE = bin_R < (nBinsRHisto_ - 1) ? histoClusters.at(z_side, bin_R + 1, binRight).sumMipPt : 0;

        isMax &= MIPT_seed >= MIPT_S && MIPT_seed > MIPT_N && MIPT_seed >= MIPT_E && MIPT_seed >= MIPT_SE &&
                 MIPT_seed >= MIPT_NE && MIPT_seed > MIPT_W && MIPT_seed > MIPT_SW && MIPT_seed > MIPT_NW;

        if (isMax) {
          setSeedEnergyAndPosition(
              seedPositionsEnergy, z_side, bin_R, bin_phi, histoClusters.at(z_side, bin_R, bin_phi));
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
    for (unsigned bin_R = 0; bin_R < nBinsRHisto_; bin_R++) {
      for (unsigned bin_phi = 0; bin_phi < nBinsPhiHisto_; bin_phi++) {
        float MIPT_seed = histoClusters.at(z_side, bin_R, bin_phi).sumMipPt;
        float MIPT_S = bin_R < (nBinsRHisto_ - 1) ? histoClusters.at(z_side, bin_R + 1, bin_phi).sumMipPt : 0;
        float MIPT_N = bin_R > 0 ? histoClusters.at(z_side, bin_R - 1, bin_phi).sumMipPt : 0;

        int binLeft = bin_phi - 1;
        if (binLeft < 0)
          binLeft += nBinsPhiHisto_;
        unsigned binRight = bin_phi + 1;
        if (binRight >= nBinsPhiHisto_)
          binRight -= nBinsPhiHisto_;

        float MIPT_W = histoClusters.at(z_side, bin_R, binLeft).sumMipPt;
        float MIPT_E = histoClusters.at(z_side, bin_R, binRight).sumMipPt;

        float MIPT_NW = bin_R > 0 ? histoClusters.at(z_side, bin_R - 1, binLeft).sumMipPt : 0;
        float MIPT_NE = bin_R > 0 ? histoClusters.at(z_side, bin_R - 1, binRight).sumMipPt : 0;
        float MIPT_SW = bin_R < (nBinsRHisto_ - 1) ? histoClusters.at(z_side, bin_R + 1, binLeft).sumMipPt : 0;
        float MIPT_SE = bin_R < (nBinsRHisto_ - 1) ? histoClusters.at(z_side, bin_R + 1, binRight).sumMipPt : 0;

        float MIPT_pred = neighbour_weights_.at(0) * MIPT_NW + neighbour_weights_.at(1) * MIPT_N +
                          neighbour_weights_.at(2) * MIPT_NE + neighbour_weights_.at(3) * MIPT_W +
                          neighbour_weights_.at(5) * MIPT_E + neighbour_weights_.at(6) * MIPT_SW +
                          neighbour_weights_.at(7) * MIPT_S + neighbour_weights_.at(8) * MIPT_SE;

        bool isMax = MIPT_seed >= (MIPT_pred + histoThreshold_);

        if (isMax) {
          setSeedEnergyAndPosition(
              seedPositionsEnergy, z_side, bin_R, bin_phi, histoClusters.at(z_side, bin_R, bin_phi));
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
    for (unsigned bin_R = 0; bin_R < nBinsRHisto_; bin_R++) {
      for (unsigned bin_phi = 0; bin_phi < nBinsPhiHisto_; bin_phi++) {
        float MIPT_seed = histoClusters.at(z_side, bin_R, bin_phi).sumMipPt;
        bool isSeed = MIPT_seed > histoThreshold_;

        if (isSeed) {
          setSeedEnergyAndPosition(
              seedPositionsEnergy, z_side, bin_R, bin_phi, histoClusters.at(z_side, bin_R, bin_phi));
        }
      }
    }
  }

  return seedPositionsEnergy;
}

std::vector<std::pair<GlobalPoint, double>> HGCalHistoSeedingImpl::computeSecondaryMaxSeeds(
    const Histogram& histoClusters) {
  std::vector<std::pair<GlobalPoint, double>> seedPositionsEnergy;

  HistogramT<uint8_t> primarySeedPositions(nBinsRHisto_, nBinsPhiHisto_);
  HistogramT<uint8_t> vetoPositions(nBinsRHisto_, nBinsPhiHisto_);

  //Search for primary seeds
  for (int z_side : {-1, 1}) {
    for (unsigned bin_R = 0; bin_R < nBinsRHisto_; bin_R++) {
      for (unsigned bin_phi = 0; bin_phi < nBinsPhiHisto_; bin_phi++) {
        float MIPT_seed = histoClusters.at(z_side, bin_R, bin_phi).sumMipPt;
        bool isMax = MIPT_seed > histoThreshold_;

        if (!isMax)
          continue;

        float MIPT_S = bin_R < (nBinsRHisto_ - 1) ? histoClusters.at(z_side, bin_R + 1, bin_phi).sumMipPt : 0;
        float MIPT_N = bin_R > 0 ? histoClusters.at(z_side, bin_R - 1, bin_phi).sumMipPt : 0;

        int binLeft = bin_phi - 1;
        if (binLeft < 0)
          binLeft += nBinsPhiHisto_;
        unsigned binRight = bin_phi + 1;
        if (binRight >= nBinsPhiHisto_)
          binRight -= nBinsPhiHisto_;

        float MIPT_W = histoClusters.at(z_side, bin_R, binLeft).sumMipPt;
        float MIPT_E = histoClusters.at(z_side, bin_R, binRight).sumMipPt;
        float MIPT_NW = bin_R > 0 ? histoClusters.at(z_side, bin_R - 1, binLeft).sumMipPt : 0;
        float MIPT_NE = bin_R > 0 ? histoClusters.at(z_side, bin_R - 1, binRight).sumMipPt : 0;
        float MIPT_SW = bin_R < (nBinsRHisto_ - 1) ? histoClusters.at(z_side, bin_R + 1, binLeft).sumMipPt : 0;
        float MIPT_SE = bin_R < (nBinsRHisto_ - 1) ? histoClusters.at(z_side, bin_R + 1, binRight).sumMipPt : 0;

        isMax &= MIPT_seed >= MIPT_S && MIPT_seed > MIPT_N && MIPT_seed >= MIPT_E && MIPT_seed >= MIPT_SE &&
                 MIPT_seed >= MIPT_NE && MIPT_seed > MIPT_W && MIPT_seed > MIPT_SW && MIPT_seed > MIPT_NW;

        if (isMax) {
          setSeedEnergyAndPosition(
              seedPositionsEnergy, z_side, bin_R, bin_phi, histoClusters.at(z_side, bin_R, bin_phi));

          primarySeedPositions.at(z_side, bin_R, bin_phi) = true;

          vetoPositions.at(z_side, bin_R, binLeft) = true;
          vetoPositions.at(z_side, bin_R, binRight) = true;
          if (bin_R > 0) {
            vetoPositions.at(z_side, bin_R - 1, bin_phi) = true;
            vetoPositions.at(z_side, bin_R - 1, binRight) = true;
            vetoPositions.at(z_side, bin_R - 1, binLeft) = true;
          }
          if (bin_R < (nBinsRHisto_ - 1)) {
            vetoPositions.at(z_side, bin_R + 1, bin_phi) = true;
            vetoPositions.at(z_side, bin_R + 1, binRight) = true;
            vetoPositions.at(z_side, bin_R + 1, binLeft) = true;
          }
        }
      }
    }
  }

  //Search for secondary seeds

  for (int z_side : {-1, 1}) {
    for (unsigned bin_R = 0; bin_R < nBinsRHisto_; bin_R++) {
      for (unsigned bin_phi = 0; bin_phi < nBinsPhiHisto_; bin_phi++) {
        //Cannot be a secondary seed if already a primary seed, or adjacent to primary seed
        if (primarySeedPositions.at(z_side, bin_R, bin_phi) || vetoPositions.at(z_side, bin_R, bin_phi))
          continue;

        float MIPT_seed = histoClusters.at(z_side, bin_R, bin_phi).sumMipPt;
        bool isMax = MIPT_seed > histoThreshold_;

        float MIPT_S = bin_R < (nBinsRHisto_ - 1) ? histoClusters.at(z_side, bin_R + 1, bin_phi).sumMipPt : 0;
        float MIPT_N = bin_R > 0 ? histoClusters.at(z_side, bin_R - 1, bin_phi).sumMipPt : 0;

        int binLeft = bin_phi - 1;
        if (binLeft < 0)
          binLeft += nBinsPhiHisto_;
        unsigned binRight = bin_phi + 1;
        if (binRight >= nBinsPhiHisto_)
          binRight -= nBinsPhiHisto_;

        float MIPT_W = histoClusters.at(z_side, bin_R, binLeft).sumMipPt;
        float MIPT_E = histoClusters.at(z_side, bin_R, binRight).sumMipPt;
        float MIPT_NW = bin_R > 0 ? histoClusters.at(z_side, bin_R - 1, binLeft).sumMipPt : 0;
        float MIPT_NE = bin_R > 0 ? histoClusters.at(z_side, bin_R - 1, binRight).sumMipPt : 0;
        float MIPT_SW = bin_R < (nBinsRHisto_ - 1) ? histoClusters.at(z_side, bin_R + 1, binLeft).sumMipPt : 0;
        float MIPT_SE = bin_R < (nBinsRHisto_ - 1) ? histoClusters.at(z_side, bin_R + 1, binRight).sumMipPt : 0;

        isMax &=
            (((bin_R < nBinsRHisto_ - 1) && vetoPositions.at(z_side, bin_R + 1, bin_phi)) or MIPT_seed >= MIPT_S) &&
            (((bin_R > 0) && vetoPositions.at(z_side, bin_R - 1, bin_phi)) or MIPT_seed > MIPT_N) &&
            ((vetoPositions.at(z_side, bin_R, binRight)) or MIPT_seed >= MIPT_E) &&
            (((bin_R < nBinsRHisto_ - 1) && vetoPositions.at(z_side, bin_R + 1, binRight)) or MIPT_seed >= MIPT_SE) &&
            (((bin_R > 0) && vetoPositions.at(z_side, bin_R - 1, binRight)) or MIPT_seed >= MIPT_NE) &&
            ((vetoPositions.at(z_side, bin_R, binLeft)) or MIPT_seed > MIPT_W) &&
            (((bin_R < nBinsRHisto_ - 1) && vetoPositions.at(z_side, bin_R + 1, binLeft)) or MIPT_seed > MIPT_SW) &&
            (((bin_R > 0) && vetoPositions.at(z_side, bin_R - 1, binLeft)) or MIPT_seed > MIPT_NW);

        if (isMax) {
          setSeedEnergyAndPosition(
              seedPositionsEnergy, z_side, bin_R, bin_phi, histoClusters.at(z_side, bin_R, bin_phi));
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

  /* smoothen along the phi direction + normalize each bin to same area */
  Histogram smoothPhiHistoCluster = fillSmoothPhiHistoClusters(histoCluster, binsSumsHisto_);

  /* smoothen along the r/z direction */
  Histogram smoothRPhiHistoCluster = fillSmoothRPhiHistoClusters(smoothPhiHistoCluster);

  /* seeds determined with local maximum criteria */
  if (seedingType_ == HistoMaxC3d)
    seedPositionsEnergy = computeMaxSeeds(smoothRPhiHistoCluster);
  else if (seedingType_ == HistoThresholdC3d)
    seedPositionsEnergy = computeThresholdSeeds(smoothRPhiHistoCluster);
  else if (seedingType_ == HistoInterpolatedMaxC3d)
    seedPositionsEnergy = computeInterpolatedMaxSeeds(smoothRPhiHistoCluster);
  else if (seedingType_ == HistoSecondaryMaxC3d)
    seedPositionsEnergy = computeSecondaryMaxSeeds(smoothRPhiHistoCluster);
}
