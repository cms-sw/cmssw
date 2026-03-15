/** Computation of input features for superclustering DNN. Used by plugins/TracksterLinkingBySuperClustering.cc and plugins/SuperclusteringSampleDumper.cc */
// Author: Theo Cuisset - theo.cuisset@cern.ch
// Date: 11/2023

// Modified by Gamze Sokmen - gamze.sokmen@cern.ch
// Changes: Implementation of the delta time feature under a new DNN input version (v3) for the superclustering DNN and correcting the seed pT calculation.
// Date: 07/2025

// Modified by Felice Pantaleo <felice.pantaleo@cern.ch>
// Improved memory usage and inference performance.
// Date: 02/2026
//

#ifndef __RecoHGCal_TICL_SuperclusteringDNNInputs_H__
#define __RecoHGCal_TICL_SuperclusteringDNNInputs_H__

#include "DataFormats/HGCalReco/interface/TracksterFwd.h"
#include <vector>
#include <string>
#include <memory>
#include <span>
#include <string>
#include <vector>

namespace ticl {

  // any raw_dt outside +/- kDeltaTimeDefault is considered bad
  inline constexpr float kDeltaTimeDefault = 50.f;
  inline constexpr float kBadDeltaTime = -5.f;

  // Abstract base class for DNN input preparation.
  class AbstractSuperclusteringDNNInput {
  public:
    virtual ~AbstractSuperclusteringDNNInput() = default;

    virtual unsigned int featureCount() const = 0;

    /** Get name of features. Used for SuperclusteringSampleDumper branch names (inference does not use the names, only the indices) */
    virtual std::vector<std::string> featureNames() const {
      std::vector<std::string> defaultNames;
      defaultNames.reserve(featureCount());
      for (unsigned int i = 1; i <= featureCount(); ++i) {
        defaultNames.emplace_back(std::string("nb_") + std::to_string(i));
      }
      return defaultNames;
    }

    /** Compute features for seed and candidate pair into user-provided buffer (size = featureCount()). */
    virtual void computeInto(Trackster const& ts_base, Trackster const& ts_toCluster, std::span<float> out) const = 0;

  protected:
    AbstractSuperclusteringDNNInput() = default;
  };

  class SuperclusteringDNNInputV1 final : public AbstractSuperclusteringDNNInput {
  public:
    static constexpr unsigned int kNFeatures = 9;
    unsigned int featureCount() const override { return kNFeatures; }

    void computeInto(Trackster const& ts_base, Trackster const& ts_toCluster, std::span<float> out) const override;

    std::vector<std::string> featureNames() const override {
      return {"DeltaEtaBaryc",
              "DeltaPhiBaryc",
              "multi_en",
              "multi_eta",
              "multi_pt",
              "seedEta",
              "seedPhi",
              "seedEn",
              "seedPt"};
    }
  };

  class SuperclusteringDNNInputV2 final : public AbstractSuperclusteringDNNInput {
  public:
    static constexpr unsigned int kNFeatures = 17;
    unsigned int featureCount() const override { return kNFeatures; }

    void computeInto(Trackster const& ts_base, Trackster const& ts_toCluster, std::span<float> out) const override;

    std::vector<std::string> featureNames() const override {
      return {"DeltaEtaBaryc",
              "DeltaPhiBaryc",
              "multi_en",
              "multi_eta",
              "multi_pt",
              "seedEta",
              "seedPhi",
              "seedEn",
              "seedPt",
              "theta",
              "theta_xz_seedFrame",
              "theta_yz_seedFrame",
              "theta_xy_cmsFrame",
              "theta_yz_cmsFrame",
              "theta_xz_cmsFrame",
              "explVar",
              "explVarRatio"};
    }
  };

  class SuperclusteringDNNInputV3 final : public AbstractSuperclusteringDNNInput {
  public:
    static constexpr unsigned int kNFeatures = 18;
    unsigned int featureCount() const override { return kNFeatures; }

    void computeInto(Trackster const& ts_base, Trackster const& ts_toCluster, std::span<float> out) const override;

    std::vector<std::string> featureNames() const override {
      return {"DeltaEtaBaryc",
              "DeltaPhiBaryc",
              "multi_en",
              "multi_eta",
              "multi_pt",
              "seedEta",
              "seedPhi",
              "seedEn",
              "seedPt",
              "theta",
              "theta_xz_seedFrame",
              "theta_yz_seedFrame",
              "theta_xy_cmsFrame",
              "theta_yz_cmsFrame",
              "theta_xz_cmsFrame",
              "explVar",
              "explVarRatio",
              "mod_deltaTime"};
    }
  };

  std::unique_ptr<AbstractSuperclusteringDNNInput> makeSuperclusteringDNNInputFromString(std::string dnnVersion);

}  // namespace ticl

#endif
