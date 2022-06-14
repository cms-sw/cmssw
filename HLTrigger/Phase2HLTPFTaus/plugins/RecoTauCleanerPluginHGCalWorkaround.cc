#include "HLTrigger/Phase2HLTPFTaus/plugins/RecoTauCleanerPluginHGCalWorkaround.h"

RecoTauCleanerPluginHGCalWorkaround::RecoTauCleanerPluginHGCalWorkaround(const edm::ParameterSet& cfg,
                                                                         edm::ConsumesCollector&& cc)
    : reco::tau::RecoTauCleanerPlugin(cfg, std::move(cc)) {}

double RecoTauCleanerPluginHGCalWorkaround::operator()(const reco::PFTauRef& pfTau) const {
    if (pfTau->leadPFChargedHadrCand().isNonnull() && pfTau->leadPFChargedHadrCand()->bestTrack()) {
        // CV: negative sign means that we prefer PFTaus with a "leading" reco::Track of high pT
        return -pfTau->leadPFChargedHadrCand()->bestTrack()->pt();
    } else {
        return 0.;
    }
}

DEFINE_EDM_PLUGIN(RecoTauCleanerPluginFactory,
                  RecoTauCleanerPluginHGCalWorkaround,
                  "RecoTauCleanerPluginHGCalWorkaround");
