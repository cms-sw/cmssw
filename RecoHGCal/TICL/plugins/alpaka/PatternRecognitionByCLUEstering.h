
#pragma once

#include "CondCore/CondDB/interface/Exception.h"
#include "DataFormats/HGCalReco/interface/HGCalSoAClusters.h"
#include "DataFormats/HGCalReco/interface/HGCalSoARecHitsHostCollection.h"
#include "DataFormats/HGCalReco/interface/alpaka/HGCalSoAClustersDeviceCollection.h"
#include "DataFormats/HGCalReco/interface/alpaka/HGCalSoARecHitsExtraDeviceCollection.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "RecoHGCal/TICL/interface/alpaka/PatternRecognitionAlgoBase.h"
#include "DataFormats/HGCalReco/interface/Trackster.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

#include <algorithm>
#include <array>
#include <ranges>
#include <unordered_map>
#include <vector>

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  class PatternRecognitionByCLUEstering final : public PatternRecognitionAlgoBase {
  private:
    float m_rhoc;
    std::array<float, 3> m_dc;
    std::array<float, 3> m_dm;

  public:
    PatternRecognitionByCLUEstering(const edm::ParameterSet& config)
        : PatternRecognitionAlgoBase(config), m_rhoc(config.getParameter<double>("rho_c")) {
      auto dc = config.getParameter<std::vector<double>>("dc");
      auto dm = config.getParameter<std::vector<double>>("dm");

      if (dc.size() != 3 || dm.size() != 3) {
        throw cms::Exception("Configuration") << "Parameters 'dc' and 'dm' must each have exactly 3 elements.";
      }

      auto to_float = [](auto x) -> float { return static_cast<float>(x); };
      std::ranges::copy(dc | std::views::transform(to_float), m_dc.begin());
      std::ranges::copy(dm | std::views::transform(to_float), m_dm.begin());
    }
    ~PatternRecognitionByCLUEstering() override = default;

    void makeTracksters(Queue& queue,
                        const HGCalSoAClustersDeviceCollection& lc,
                        std::vector<ticl::Trackster>& tracksters) override;

    static void fillPSetDescription(::edm::ParameterSetDescription& iDesc);
  };

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE
