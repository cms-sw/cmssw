int jet_flavour(const pat::Jet& jet,
                const std::vector<reco::GenParticle>& gToBB,
                const std::vector<reco::GenParticle>& gToCC,
                const std::vector<reco::GenParticle>& neutrinosLepB,
                const std::vector<reco::GenParticle>& neutrinosLepB_C,
                const std::vector<reco::GenParticle>& alltaus,
                bool usePhysForLightAndUndefined);
