#ifndef HLTrigger_Phase2HLTPFTaus_RecoTauCleanerPluginHGCalWorkaround_h
#define HLTrigger_Phase2HLTPFTaus_RecoTauCleanerPluginHGCalWorkaround_h

#include "RecoTauTag/RecoTau/interface/RecoTauBuilderPlugins.h"
#include "DataFormats/TauReco/interface/PFTau.h"
#include "DataFormats/TauReco/interface/PFTauFwd.h"

class RecoTauCleanerPluginHGCalWorkaround : public reco::tau::RecoTauCleanerPlugin {
 public:
    explicit RecoTauCleanerPluginHGCalWorkaround(const edm::ParameterSet& cfg, edm::ConsumesCollector&& cc);
    ~RecoTauCleanerPluginHGCalWorkaround() override = default;

    double operator()(const reco::PFTauRef& pfTau) const override;
};

#endif
