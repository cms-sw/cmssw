// Author: Felice Pantaleo,Marco Rovere - felice.pantaleo@cern.ch, marco.rovere@cern.ch
// Date: 09/2018

#ifndef __RecoHGCal_TICL_PRbyCA_H__
#define __RecoHGCal_TICL_PRbyCA_H__
#include <algorithm>
#include <iostream>
#include "DataFormats/Math/interface/normalizedPhi.h"
#include "PatternRecognitionbyCAConstants.h"
#include "HGCDoublet.h"
#include "HGCGraph.h"
#include "RecoHGCal/TICL/interface/PatternRecognitionAlgoBase.h"
#include "RecoHGCal/TICL/interface/Common.h"
#include "RecoLocalCalo/HGCalRecAlgos/interface/RecHitTools.h"

namespace ticl {
  class PatternRecognitionbyCA final : public PatternRecognitionAlgoBase {
    public:
      PatternRecognitionbyCA(const edm::ParameterSet& conf) : PatternRecognitionAlgoBase(conf) {
        min_cos_theta_ = (float)conf.getParameter<double>("min_cos_theta");
        min_cos_pointing_ = (float)conf.getParameter<double>("min_cos_pointing");
        missing_layers_ = conf.getParameter<int>("missing_layers");
        min_clusters_per_ntuplet_ = conf.getParameter<int>("min_clusters_per_ntuplet");
      }

      void fillHistogram(const std::vector<reco::CaloCluster>& layerClusters,
          const HgcalClusterFilterMask& mask);

      void makeTracksters(const edm::Event& ev, const edm::EventSetup& es,
          const std::vector<reco::CaloCluster>& layerClusters,
          const HgcalClusterFilterMask& mask,
          std::vector<Trackster>& result) override;

    private:
      int getEtaBin(float eta) const {
        constexpr float etaRange = ticl::constants::maxEta - ticl::constants::minEta;
        static_assert(etaRange >= 0.f);
        float r = patternbyca::nEtaBins / etaRange;
        int etaBin = (std::abs(eta) - ticl::constants::minEta) * r;
        etaBin = std::clamp(etaBin, 0, patternbyca::nEtaBins);
        return etaBin;
      }

      int getPhiBin(float phi) const {
        auto normPhi = normalizedPhi(phi);
        float r = patternbyca::nPhiBins * M_1_PI * 0.5f;
        int phiBin = (normPhi + M_PI) * r;

        return phiBin;
      }

      int globalBin(int etaBin, int phiBin) const { return phiBin + etaBin * patternbyca::nPhiBins; }

      void clearHistogram() {
        auto nBins = patternbyca::nEtaBins * patternbyca::nPhiBins;
        for (int i = 0; i < patternbyca::nLayers; ++i) {
          for (int j = 0; j < nBins; ++j) tile_[i][j].clear();
        }
      }


      hgcal::RecHitTools rhtools_;
      patternbyca::Tile tile_;  // a histogram of layerClusters IDs per layer
      HGCGraph theGraph_;
      float min_cos_theta_;
      float min_cos_pointing_;
      int missing_layers_;
      int min_clusters_per_ntuplet_;
  };
}
#endif
