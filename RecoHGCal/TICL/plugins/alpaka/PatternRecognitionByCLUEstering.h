
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
    float m_dc;
    float m_dm;

  public:
    PatternRecognitionByCLUEstering(const edm::ParameterSet& config)
        : PatternRecognitionAlgoBase(config),
          m_rhoc(config.getParameter<double>("rho_c")),
          m_dc(config.getParameter<double>("dc")),
          m_dm(config.getParameter<double>("dm")) {}
    ~PatternRecognitionByCLUEstering() override = default;

    void makeTracksters(Queue& queue,
                        const HGCalSoAClustersDeviceCollection& lc,
                        std::vector<ticl::Trackster>& tracksters) override;

    static void fillPSetDescription(::edm::ParameterSetDescription& iDesc);
  };

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE
