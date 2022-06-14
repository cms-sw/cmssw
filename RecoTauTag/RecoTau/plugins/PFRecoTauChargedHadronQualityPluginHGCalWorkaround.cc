#include "RecoTauTag/RecoTau/interface/PFRecoTauChargedHadronPlugins.h"
#include "DataFormats/TauReco/interface/PFRecoTauChargedHadron.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"

class PFRecoTauChargedHadronQualityPluginHGCalWorkaround : public reco::tau::PFRecoTauChargedHadronQualityPlugin {
public:
    explicit PFRecoTauChargedHadronQualityPluginHGCalWorkaround(const edm::ParameterSet& cfg);
    ~PFRecoTauChargedHadronQualityPluginHGCalWorkaround() override = default;

    double operator()(const reco::PFRecoTauChargedHadron& pfTauChargedHadron) const override;
};

PFRecoTauChargedHadronQualityPluginHGCalWorkaround::PFRecoTauChargedHadronQualityPluginHGCalWorkaround(
                                                                                                       const edm::ParameterSet& cfg)
    : reco::tau::PFRecoTauChargedHadronQualityPlugin(cfg) {}

double PFRecoTauChargedHadronQualityPluginHGCalWorkaround::operator()(
                                                                      const reco::PFRecoTauChargedHadron& pfTauChargedHadron) const {
    if (pfTauChargedHadron.algo() == reco::PFRecoTauChargedHadron::kChargedPFCandidate) {
        const reco::PFCandidate* chargedPFCand =
            dynamic_cast<const reco::PFCandidate*>(pfTauChargedHadron.getChargedPFCandidate().get());
        assert(chargedPFCand);
        if (chargedPFCand->bestTrack()) {
            // CV: negative sign means that we prefer PFRecoTauChargedHadrons with a reco::Track of high pT
            return -chargedPFCand->bestTrack()->pt();
        }
    }
    return 0.;
}

#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_EDM_PLUGIN(PFRecoTauChargedHadronQualityPluginFactory,
                  PFRecoTauChargedHadronQualityPluginHGCalWorkaround,
                  "PFRecoTauChargedHadronQualityPluginHGCalWorkaround");
