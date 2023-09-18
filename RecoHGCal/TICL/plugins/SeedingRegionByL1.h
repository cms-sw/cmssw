// Author: Swagata Mukherjee

#ifndef RecoHGCal_TICL_SeedingRegionByL1_h
#define RecoHGCal_TICL_SeedingRegionByL1_h
#include <memory>
#include <string>

#include "DataFormats/L1TCorrelator/interface/TkEm.h"
#include "DataFormats/L1TCorrelator/interface/TkEmFwd.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/ESGetToken.h"
#include "RecoHGCal/TICL/plugins/SeedingRegionAlgoBase.h"

namespace ticl {
  class SeedingRegionByL1 final : public SeedingRegionAlgoBase {
  public:
    SeedingRegionByL1(const edm::ParameterSet& conf, edm::ConsumesCollector& sumes);

    void initialize(const edm::EventSetup& es) override{};
    void makeRegions(const edm::Event& ev, const edm::EventSetup& es, std::vector<TICLSeedingRegion>& result) override;
    static void fillPSetDescription(edm::ParameterSetDescription& desc);

  private:
    edm::EDGetTokenT<std::vector<l1t::TkEm>> l1TkEmsToken_;
    int algoVerbosity_ = 0;
    double minPt_;                        // minimum pT of L1 TkEm objects
    double minAbsEta_;                    // minimum |eta| of L1 TkEm objects
    double maxAbsEta_;                    // maximum |eta| of L1 TkEm objects
    std::vector<double> endcapScalings_;  // pT scaling factors for endcap
    int quality_;                         // hwQual
    bool qualityIsMask_;                  // is hwQual to be applied as a mask
    bool applyQuality_;                   // whether to apply quality

    double tkEmOfflineEt(double et) const;
  };
}  // namespace ticl
#endif
