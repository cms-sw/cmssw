#ifndef L1Trigger_Phase2L1GT_L1GTObjectConfig_h
#define L1Trigger_Phase2L1GT_L1GTObjectConfig_h

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "L1Trigger/Phase2L1GT/interface/L1GTOfflineThreshold.h"

#include <string>
#include <vector>

/*
 * Resolution of the structured collection configuration used by the P2GT menus:
 *
 *   collectionN = cms.PSet(
 *       object = cms.PSet(refToPSet_ = cms.string("l1tGTtkMuonVLoose")),  # shared object definition
 *       ptScaling = cms.PSet(refToPSet_ = cms.string("...")),             # optional, overrides object.ptScaling
 *       offlineMinPt = cms.double(22),                                     # offline threshold
 *       ...                                                                # any cut, overrides the object value
 *   )
 *
 * refToPSet_ is resolved by the python configuration layer, so here 'object' and
 * 'ptScaling' are ordinary nested PSets. The resolved configuration is the flat
 * form understood by L1GTSingleCollectionCut: object parameters merged with the
 * collection-level ones (the latter win) and offline thresholds converted to online
 * thresholds with the object scaling:
 *   offlineMinPt          -> regionsMinPt (multiple scaling regions) or minPt (single region)
 *   offlineMinScalarSumPt -> minScalarSumPt (single region)
 */

namespace l1t {

  inline std::vector<double> offlineToOnlineThresholds(double offline, const edm::ParameterSet& scaling) {
    const auto& offsets = scaling.getParameter<std::vector<double>>("regionsOffset");
    const auto& slopes = scaling.getParameter<std::vector<double>>("regionsSlope");
    const double minOnline = scaling.existsAs<double>("minOnlinePt") ? scaling.getParameter<double>("minOnlinePt") : 0.;

    if (offsets.empty() || offsets.size() != slopes.size()) {
      throw cms::Exception("Configuration") << "'ptScaling' needs 'regionsOffset' and 'regionsSlope' of equal, "
                                            << "non-zero length (got " << offsets.size() << " and " << slopes.size()
                                            << ").";
    }

    std::vector<double> thresholds(offsets.size());
    for (std::size_t i = 0; i < offsets.size(); ++i) {
      thresholds[i] = offlineToOnlinePt(offline, offsets[i], slopes[i], minOnline);
    }
    return thresholds;
  }

  inline edm::ParameterSet resolveCollectionConfig(const edm::ParameterSet& config) {
    edm::ParameterSet resolved(config);

    if (config.existsAs<edm::ParameterSet>("object")) {
      const edm::ParameterSet& object = config.getParameterSet("object");
      for (const std::string& name : object.getParameterNames()) {
        if (!resolved.exists(name)) {
          resolved.copyFrom(object, name);
        }
      }
    }

    const bool hasOfflineMinPt = resolved.existsAs<double>("offlineMinPt");
    const bool hasOfflineMinScalarSumPt = resolved.existsAs<double>("offlineMinScalarSumPt");
    if (!hasOfflineMinPt && !hasOfflineMinScalarSumPt) {
      return resolved;
    }

    if (!resolved.existsAs<edm::ParameterSet>("ptScaling")) {
      throw cms::Exception("Configuration")
          << "An offline threshold is given but neither the collection nor its object defines 'ptScaling'.";
    }
    const edm::ParameterSet& scaling = resolved.getParameterSet("ptScaling");
    const std::size_t nEtaRegions = resolved.existsAs<std::vector<double>>("regionsAbsEtaLowerBounds")
                                        ? resolved.getParameter<std::vector<double>>("regionsAbsEtaLowerBounds").size()
                                        : 0;

    if (hasOfflineMinPt) {
      const std::vector<double> thresholds =
          offlineToOnlineThresholds(resolved.getParameter<double>("offlineMinPt"), scaling);
      const std::string target = thresholds.size() > 1 ? "regionsMinPt" : "minPt";
      if (resolved.exists(target)) {
        throw cms::Exception("Configuration") << "'offlineMinPt' and '" << target << "' are mutually exclusive.";
      }
      if (thresholds.size() > 1 && thresholds.size() != nEtaRegions) {
        throw cms::Exception("Configuration")
            << "'ptScaling' has " << thresholds.size() << " regions, but 'regionsAbsEtaLowerBounds' has " << nEtaRegions
            << ".";
      }
      if (thresholds.size() > 1) {
        resolved.addParameter<std::vector<double>>(target, thresholds);
      } else {
        resolved.addParameter<double>(target, thresholds.front());
      }
    }

    if (hasOfflineMinScalarSumPt) {
      const std::vector<double> thresholds =
          offlineToOnlineThresholds(resolved.getParameter<double>("offlineMinScalarSumPt"), scaling);
      if (thresholds.size() != 1) {
        throw cms::Exception("Configuration") << "'offlineMinScalarSumPt' requires a 'ptScaling' with a single region.";
      }
      if (resolved.exists("minScalarSumPt")) {
        throw cms::Exception("Configuration") << "'offlineMinScalarSumPt' and 'minScalarSumPt' are mutually exclusive.";
      }
      resolved.addParameter<double>("minScalarSumPt", thresholds.front());
    }

    return resolved;
  }

  // Input tag of a (possibly structured) collection configuration, without resolving thresholds.
  inline edm::InputTag collectionTag(const edm::ParameterSet& config) {
    if (config.existsAs<edm::InputTag>("tag")) {
      return config.getParameter<edm::InputTag>("tag");
    }
    if (config.existsAs<edm::ParameterSet>("object")) {
      return config.getParameterSet("object").getParameter<edm::InputTag>("tag");
    }
    throw cms::Exception("Configuration") << "Collection configuration has neither 'tag' nor 'object.tag'.";
  }

  inline void fillPtScalingDescription(edm::ParameterSetDescription& desc) {
    edm::ParameterSetDescription scalingDesc;
    scalingDesc.add<std::vector<double>>("regionsOffset");
    scalingDesc.add<std::vector<double>>("regionsSlope");
    scalingDesc.addOptional<double>("minOnlinePt");
    desc.addOptional<edm::ParameterSetDescription>("ptScaling", scalingDesc);
  }

}  // namespace l1t

#endif  // L1Trigger_Phase2L1GT_L1GTObjectConfig_h
