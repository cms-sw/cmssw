/** Computation of input features for superclustering DNN. Used by plugins/TracksterLinkingBySuperClustering.cc and plugins/SuperclusteringSampleDumper.cc */
// Author: Theo Cuisset - theo.cuisset@cern.ch
// Date: 11/2023

#ifndef __RecoHGCal_TICL_SuperclusteringDNNInputs_H__
#define __RecoHGCal_TICL_SuperclusteringDNNInputs_H__

#include <vector>
#include <string>
#include <memory>

namespace ticl {
  class Trackster;

  // Abstract base class for DNN input preparation.
  class AbstractSuperclusteringDNNInput {
  public:
    virtual ~AbstractSuperclusteringDNNInput() = default;

    virtual unsigned int featureCount() const { return featureNames().size(); };

    /** Get name of features. Used for SuperclusteringSampleDumper branch names (inference does not use the names, only the indices) 
     * The default implementation is meant to be overriden by inheriting classes
    */
    virtual std::vector<std::string> featureNames() const {
      std::vector<std::string> defaultNames;
      defaultNames.reserve(featureCount());
      for (unsigned int i = 1; i <= featureCount(); i++) {
        defaultNames.push_back(std::string("nb_") + std::to_string(i));
      }
      return defaultNames;
    }

    /** Compute feature for seed and candidate pair */
    virtual std::vector<float> computeVector(ticl::Trackster const& ts_base, ticl::Trackster const& ts_toCluster) = 0;
  };

  /* First version of DNN by Alessandro Tarabini. Meant as a DNN equivalent of Mustache algorithm (superclustering algo in ECAL)
  Uses features : ['DeltaEta', 'DeltaPhi', 'multi_en', 'multi_eta', 'multi_pt', 'seedEta','seedPhi','seedEn', 'seedPt']
  */
  class SuperclusteringDNNInputV1 : public AbstractSuperclusteringDNNInput {
  public:
    unsigned int featureCount() const override { return 9; }

    std::vector<float> computeVector(ticl::Trackster const& ts_base, ticl::Trackster const& ts_toCluster) override;

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

  /* Second version of DNN by Alessandro Tarabini, making use of HGCAL-specific features.
  Uses features : ['DeltaEta', 'DeltaPhi', 'multi_en', 'multi_eta', 'multi_pt', 'seedEta','seedPhi','seedEn', 'seedPt', 'theta', 'theta_xz_seedFrame', 'theta_yz_seedFrame', 'theta_xy_cmsFrame', 'theta_yz_cmsFrame', 'theta_xz_cmsFrame', 'explVar', 'explVarRatio']
  */
  class SuperclusteringDNNInputV2 : public AbstractSuperclusteringDNNInput {
  public:
    unsigned int featureCount() const override { return 17; }

    std::vector<float> computeVector(ticl::Trackster const& ts_base, ticl::Trackster const& ts_toCluster) override;

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

  std::unique_ptr<AbstractSuperclusteringDNNInput> makeSuperclusteringDNNInputFromString(std::string dnnVersion);
}  // namespace ticl

#endif