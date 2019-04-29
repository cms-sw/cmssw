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
  Histogram histoClusters;  //key[0] = z.side(), key[1] = bin_R, key[2] = bin_phi

  for (int z_side : {-1, 1}) {
    for (int bin_R = 0; bin_R < int(nBinsRHisto_); bin_R++) {
      for (int bin_phi = 0; bin_phi < int(nBinsPhiHisto_); bin_phi++) {
        histoClusters[{{z_side, bin_R, bin_phi}}] = 0;
      }
    }
  }

  for (auto& clu : clustersPtrs) {
    float ROverZ = sqrt(pow(clu->centreProj().x(), 2) + pow(clu->centreProj().y(), 2));
    int bin_R = int((ROverZ - kROverZMin_) * nBinsRHisto_ / (kROverZMax_ - kROverZMin_));
    int bin_phi = int((reco::reduceRange(clu->phi()) + M_PI) * nBinsPhiHisto_ / (2 * M_PI));

    histoClusters[{{triggerTools_.zside(clu->detId()), bin_R, bin_phi}}] += clu->mipPt();
  }

  return histoClusters;
}

HGCalHistoSeedingImpl::Histogram HGCalHistoSeedingImpl::fillSmoothPhiHistoClusters(const Histogram& histoClusters,
                                                                                   const vector<unsigned>& binSums) {
  Histogram histoSumPhiClusters;  //key[0] = z.side(), key[1] = bin_R, key[2] = bin_phi

  for (int z_side : {-1, 1}) {
    for (int bin_R = 0; bin_R < int(nBinsRHisto_); bin_R++) {
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

      for (int bin_phi = 0; bin_phi < int(nBinsPhiHisto_); bin_phi++) {
        float content = histoClusters.at({{z_side, bin_R, bin_phi}});

        for (int bin_phi2 = 1; bin_phi2 <= nBinsSide; bin_phi2++) {
          int binToSumLeft = bin_phi - bin_phi2;
          if (binToSumLeft < 0)
            binToSumLeft += nBinsPhiHisto_;
          int binToSumRight = bin_phi + bin_phi2;
          if (binToSumRight >= int(nBinsPhiHisto_))
            binToSumRight -= nBinsPhiHisto_;

          content += histoClusters.at({{z_side, bin_R, binToSumLeft}}) / pow(2, bin_phi2);   // quadratic kernel
          content += histoClusters.at({{z_side, bin_R, binToSumRight}}) / pow(2, bin_phi2);  // quadratic kernel
        }

        histoSumPhiClusters[{{z_side, bin_R, bin_phi}}] = content / area;
      }
    }
  }

  return histoSumPhiClusters;
}

HGCalHistoSeedingImpl::Histogram HGCalHistoSeedingImpl::fillSmoothRPhiHistoClusters(const Histogram& histoClusters) {
  Histogram histoSumRPhiClusters;  //key[0] = z.side(), key[1] = bin_R, key[2] = bin_phi

  for (int z_side : {-1, 1}) {
    for (int bin_R = 0; bin_R < int(nBinsRHisto_); bin_R++) {
      float weight = (bin_R == 0 || bin_R == int(nBinsRHisto_) - 1)
                         ? 1.5
                         : 2.;  //Take into account edges with only one side up or down

      for (int bin_phi = 0; bin_phi < int(nBinsPhiHisto_); bin_phi++) {
        float content = histoClusters.at({{z_side, bin_R, bin_phi}});
        float contentDown = bin_R > 0 ? histoClusters.at({{z_side, bin_R - 1, bin_phi}}) : 0;
        float contentUp = bin_R < (int(nBinsRHisto_) - 1) ? histoClusters.at({{z_side, bin_R + 1, bin_phi}}) : 0;

        histoSumRPhiClusters[{{z_side, bin_R, bin_phi}}] = (content + 0.5 * contentDown + 0.5 * contentUp) / weight;
      }
    }
  }

  return histoSumRPhiClusters;
}

std::vector<std::pair<GlobalPoint, double>> HGCalHistoSeedingImpl::computeMaxSeeds(const Histogram& histoClusters) {
  std::vector<std::pair<GlobalPoint, double>> seedPositionsEnergy;

  for (int z_side : {-1, 1}) {
    for (int bin_R = 0; bin_R < int(nBinsRHisto_); bin_R++) {
      for (int bin_phi = 0; bin_phi < int(nBinsPhiHisto_); bin_phi++) {
        float MIPT_seed = histoClusters.at({{z_side, bin_R, bin_phi}});
        bool isMax = MIPT_seed > histoThreshold_;
        if (!isMax)
          continue;

        float MIPT_S = bin_R < (int(nBinsRHisto_) - 1) ? histoClusters.at({{z_side, bin_R + 1, bin_phi}}) : 0;
        float MIPT_N = bin_R > 0 ? histoClusters.at({{z_side, bin_R - 1, bin_phi}}) : 0;

        int binLeft = bin_phi - 1;
        if (binLeft < 0)
          binLeft += nBinsPhiHisto_;
        int binRight = bin_phi + 1;
        if (binRight >= int(nBinsPhiHisto_))
          binRight -= nBinsPhiHisto_;

        float MIPT_W = histoClusters.at({{z_side, bin_R, binLeft}});
        float MIPT_E = histoClusters.at({{z_side, bin_R, binRight}});
        float MIPT_NW = bin_R > 0 ? histoClusters.at({{z_side, bin_R - 1, binLeft}}) : 0;
        float MIPT_NE = bin_R > 0 ? histoClusters.at({{z_side, bin_R - 1, binRight}}) : 0;
        float MIPT_SW = bin_R < (int(nBinsRHisto_) - 1) ? histoClusters.at({{z_side, bin_R + 1, binLeft}}) : 0;
        float MIPT_SE = bin_R < (int(nBinsRHisto_) - 1) ? histoClusters.at({{z_side, bin_R + 1, binRight}}) : 0;

        isMax &= MIPT_seed >= MIPT_S && MIPT_seed > MIPT_N && MIPT_seed >= MIPT_E && MIPT_seed >= MIPT_SE &&
                 MIPT_seed >= MIPT_NE && MIPT_seed > MIPT_W && MIPT_seed > MIPT_SW && MIPT_seed > MIPT_NW;

        if (isMax) {
          float ROverZ_seed = kROverZMin_ + (bin_R + 0.5) * (kROverZMax_ - kROverZMin_) / nBinsRHisto_;
          float phi_seed = -M_PI + (bin_phi + 0.5) * 2 * M_PI / nBinsPhiHisto_;
          float x_seed = ROverZ_seed * cos(phi_seed);
          float y_seed = ROverZ_seed * sin(phi_seed);
          seedPositionsEnergy.emplace_back(GlobalPoint(x_seed, y_seed, z_side), MIPT_seed);
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
    for (int bin_R = 0; bin_R < int(nBinsRHisto_); bin_R++) {
      for (int bin_phi = 0; bin_phi < int(nBinsPhiHisto_); bin_phi++) {
        float MIPT_seed = histoClusters.at({{z_side, bin_R, bin_phi}});
        float MIPT_S = bin_R < (int(nBinsRHisto_) - 1) ? histoClusters.at({{z_side, bin_R + 1, bin_phi}}) : 0;
        float MIPT_N = bin_R > 0 ? histoClusters.at({{z_side, bin_R - 1, bin_phi}}) : 0;

        int binLeft = bin_phi - 1;
        if (binLeft < 0)
          binLeft += nBinsPhiHisto_;
        int binRight = bin_phi + 1;
        if (binRight >= int(nBinsPhiHisto_))
          binRight -= nBinsPhiHisto_;

        float MIPT_W = histoClusters.at({{z_side, bin_R, binLeft}});
        float MIPT_E = histoClusters.at({{z_side, bin_R, binRight}});

        float MIPT_NW = bin_R > 0 ? histoClusters.at({{z_side, bin_R - 1, binLeft}}) : 0;
        float MIPT_NE = bin_R > 0 ? histoClusters.at({{z_side, bin_R - 1, binRight}}) : 0;
        float MIPT_SW = bin_R < (int(nBinsRHisto_) - 1) ? histoClusters.at({{z_side, bin_R + 1, binLeft}}) : 0;
        float MIPT_SE = bin_R < (int(nBinsRHisto_) - 1) ? histoClusters.at({{z_side, bin_R + 1, binRight}}) : 0;

        float MIPT_pred = neighbour_weights_.at(0) * MIPT_NW + neighbour_weights_.at(1) * MIPT_N +
                          neighbour_weights_.at(2) * MIPT_NE + neighbour_weights_.at(3) * MIPT_W +
                          neighbour_weights_.at(5) * MIPT_E + neighbour_weights_.at(6) * MIPT_SW +
                          neighbour_weights_.at(7) * MIPT_S + neighbour_weights_.at(8) * MIPT_SE;

        bool isMax = MIPT_seed >= (MIPT_pred + histoThreshold_);

        if (isMax) {
          float ROverZ_seed = kROverZMin_ + (bin_R + 0.5) * (kROverZMax_ - kROverZMin_) / nBinsRHisto_;
          float phi_seed = -M_PI + (bin_phi + 0.5) * 2 * M_PI / nBinsPhiHisto_;
          float x_seed = ROverZ_seed * cos(phi_seed);
          float y_seed = ROverZ_seed * sin(phi_seed);
          seedPositionsEnergy.emplace_back(GlobalPoint(x_seed, y_seed, z_side), MIPT_seed);
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
    for (int bin_R = 0; bin_R < int(nBinsRHisto_); bin_R++) {
      for (int bin_phi = 0; bin_phi < int(nBinsPhiHisto_); bin_phi++) {
        float MIPT_seed = histoClusters.at({{z_side, bin_R, bin_phi}});
        bool isSeed = MIPT_seed > histoThreshold_;

        if (isSeed) {
          float ROverZ_seed = kROverZMin_ + (bin_R + 0.5) * (kROverZMax_ - kROverZMin_) / nBinsRHisto_;
          float phi_seed = -M_PI + (bin_phi + 0.5) * 2 * M_PI / nBinsPhiHisto_;
          float x_seed = ROverZ_seed * cos(phi_seed);
          float y_seed = ROverZ_seed * sin(phi_seed);
          seedPositionsEnergy.emplace_back(GlobalPoint(x_seed, y_seed, z_side), MIPT_seed);
        }
      }
    }
  }

  return seedPositionsEnergy;
}

std::vector<std::pair<GlobalPoint, double>> HGCalHistoSeedingImpl::computeSecondaryMaxSeeds(
    const Histogram& histoClusters) {
  std::vector<std::pair<GlobalPoint, double>> seedPositionsEnergy;

  std::map<std::tuple<int, int, int>, bool> primarySeedPositions;
  std::map<std::tuple<int, int, int>, bool> secondarySeedPositions;
  std::map<std::tuple<int, int, int>, bool> vetoPositions;

  //Search for primary seeds
  for (int z_side : {-1, 1}) {
    for (int bin_R = 0; bin_R < int(nBinsRHisto_); bin_R++) {
      for (int bin_phi = 0; bin_phi < int(nBinsPhiHisto_); bin_phi++) {
        float MIPT_seed = histoClusters.at({{z_side, bin_R, bin_phi}});
        bool isMax = MIPT_seed > histoThreshold_;

        if (!isMax)
          continue;

        float MIPT_S = bin_R < (int(nBinsRHisto_) - 1) ? histoClusters.at({{z_side, bin_R + 1, bin_phi}}) : 0;
        float MIPT_N = bin_R > 0 ? histoClusters.at({{z_side, bin_R - 1, bin_phi}}) : 0;

        int binLeft = bin_phi - 1;
        if (binLeft < 0)
          binLeft += nBinsPhiHisto_;
        int binRight = bin_phi + 1;
        if (binRight >= int(nBinsPhiHisto_))
          binRight -= nBinsPhiHisto_;

        float MIPT_W = histoClusters.at({{z_side, bin_R, binLeft}});
        float MIPT_E = histoClusters.at({{z_side, bin_R, binRight}});
        float MIPT_NW = bin_R > 0 ? histoClusters.at({{z_side, bin_R - 1, binLeft}}) : 0;
        float MIPT_NE = bin_R > 0 ? histoClusters.at({{z_side, bin_R - 1, binRight}}) : 0;
        float MIPT_SW = bin_R < (int(nBinsRHisto_) - 1) ? histoClusters.at({{z_side, bin_R + 1, binLeft}}) : 0;
        float MIPT_SE = bin_R < (int(nBinsRHisto_) - 1) ? histoClusters.at({{z_side, bin_R + 1, binRight}}) : 0;

        isMax &= MIPT_seed >= MIPT_S && MIPT_seed > MIPT_N && MIPT_seed >= MIPT_E && MIPT_seed >= MIPT_SE &&
                 MIPT_seed >= MIPT_NE && MIPT_seed > MIPT_W && MIPT_seed > MIPT_SW && MIPT_seed > MIPT_NW;

        if (isMax) {
          float ROverZ_seed = kROverZMin_ + (bin_R + 0.5) * (kROverZMax_ - kROverZMin_) / nBinsRHisto_;
          float phi_seed = -M_PI + (bin_phi + 0.5) * 2 * M_PI / nBinsPhiHisto_;
          float x_seed = ROverZ_seed * cos(phi_seed);
          float y_seed = ROverZ_seed * sin(phi_seed);

          seedPositionsEnergy.emplace_back(GlobalPoint(x_seed, y_seed, z_side), MIPT_seed);
          primarySeedPositions[std::make_tuple(bin_R, bin_phi, z_side)] = true;

          vetoPositions[std::make_tuple(bin_R, binLeft, z_side)] = true;
          vetoPositions[std::make_tuple(bin_R, binRight, z_side)] = true;
          if (bin_R > 0) {
            vetoPositions[std::make_tuple(bin_R - 1, bin_phi, z_side)] = true;
            vetoPositions[std::make_tuple(bin_R - 1, binRight, z_side)] = true;
            vetoPositions[std::make_tuple(bin_R - 1, binLeft, z_side)] = true;
          }
          if (bin_R < (int(nBinsRHisto_) - 1)) {
            vetoPositions[std::make_tuple(bin_R + 1, bin_phi, z_side)] = true;
            vetoPositions[std::make_tuple(bin_R + 1, binRight, z_side)] = true;
            vetoPositions[std::make_tuple(bin_R + 1, binLeft, z_side)] = true;
          }
        }
      }
    }
  }

  //Search for secondary seeds

  for (int z_side : {-1, 1}) {
    for (int bin_R = 0; bin_R < int(nBinsRHisto_); bin_R++) {
      for (int bin_phi = 0; bin_phi < int(nBinsPhiHisto_); bin_phi++) {
        //Cannot be a secondary seed if already a primary seed, or adjacent to primary seed
        if (primarySeedPositions[std::make_tuple(bin_R, bin_phi, z_side)] ||
            vetoPositions[std::make_tuple(bin_R, bin_phi, z_side)])
          continue;

        float MIPT_seed = histoClusters.at({{z_side, bin_R, bin_phi}});
        bool isMax = MIPT_seed > histoThreshold_;

        float MIPT_S = bin_R < (int(nBinsRHisto_) - 1) ? histoClusters.at({{z_side, bin_R + 1, bin_phi}}) : 0;
        float MIPT_N = bin_R > 0 ? histoClusters.at({{z_side, bin_R - 1, bin_phi}}) : 0;

        int binLeft = bin_phi - 1;
        if (binLeft < 0)
          binLeft += nBinsPhiHisto_;
        int binRight = bin_phi + 1;
        if (binRight >= int(nBinsPhiHisto_))
          binRight -= nBinsPhiHisto_;

        float MIPT_W = histoClusters.at({{z_side, bin_R, binLeft}});
        float MIPT_E = histoClusters.at({{z_side, bin_R, binRight}});
        float MIPT_NW = bin_R > 0 ? histoClusters.at({{z_side, bin_R - 1, binLeft}}) : 0;
        float MIPT_NE = bin_R > 0 ? histoClusters.at({{z_side, bin_R - 1, binRight}}) : 0;
        float MIPT_SW = bin_R < (int(nBinsRHisto_) - 1) ? histoClusters.at({{z_side, bin_R + 1, binLeft}}) : 0;
        float MIPT_SE = bin_R < (int(nBinsRHisto_) - 1) ? histoClusters.at({{z_side, bin_R + 1, binRight}}) : 0;

        isMax &= (vetoPositions[std::make_tuple(bin_R + 1, bin_phi, z_side)] or MIPT_seed >= MIPT_S) &&
                 (vetoPositions[std::make_tuple(bin_R - 1, bin_phi, z_side)] or MIPT_seed > MIPT_N) &&
                 (vetoPositions[std::make_tuple(bin_R, binRight, z_side)] or MIPT_seed >= MIPT_E) &&
                 (vetoPositions[std::make_tuple(bin_R + 1, binRight, z_side)] or MIPT_seed >= MIPT_SE) &&
                 (vetoPositions[std::make_tuple(bin_R - 1, binRight, z_side)] or MIPT_seed >= MIPT_NE) &&
                 (vetoPositions[std::make_tuple(bin_R, binLeft, z_side)] or MIPT_seed > MIPT_W) &&
                 (vetoPositions[std::make_tuple(bin_R + 1, binLeft, z_side)] or MIPT_seed > MIPT_SW) &&
                 (vetoPositions[std::make_tuple(bin_R - 1, binLeft, z_side)] or MIPT_seed > MIPT_NW);

        if (isMax) {
          float ROverZ_seed = kROverZMin_ + (bin_R + 0.5) * (kROverZMax_ - kROverZMin_) / nBinsRHisto_;
          float phi_seed = -M_PI + (bin_phi + 0.5) * 2 * M_PI / nBinsPhiHisto_;
          float x_seed = ROverZ_seed * cos(phi_seed);
          float y_seed = ROverZ_seed * sin(phi_seed);
          seedPositionsEnergy.emplace_back(GlobalPoint(x_seed, y_seed, z_side), MIPT_seed);
          secondarySeedPositions[std::make_tuple(bin_R, bin_phi, z_side)] = true;
        }
      }
    }
  }

  return seedPositionsEnergy;
}

void HGCalHistoSeedingImpl::findHistoSeeds(const std::vector<edm::Ptr<l1t::HGCalCluster>>& clustersPtrs,
                                           std::vector<std::pair<GlobalPoint, double>>& seedPositionsEnergy) {
  /* put clusters into an r/z x phi histogram */
  Histogram histoCluster = fillHistoClusters(
      clustersPtrs);  //key[0] = z.side(), key[1] = bin_R, key[2] = bin_phi, content = MIPTs summed along depth

  /* smoothen along the phi direction + normalize each bin to same area */
  Histogram smoothPhiHistoCluster = fillSmoothPhiHistoClusters(histoCluster, binsSumsHisto_);

  /* smoothen along the r/z direction */
  Histogram smoothRPhiHistoCluster = fillSmoothRPhiHistoClusters(histoCluster);

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
