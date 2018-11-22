#include "RecoEgamma/EgammaTools/interface/MVAVariableHelper.h"

/////////////
// Specializations for the Electrons
/////////////

#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"

template<>
MVAVariableHelper<reco::GsfElectron>::MVAVariableHelper(edm::ConsumesCollector && cc)
    : tokens_({
            cc.consumes<edm::ValueMap<float>>(edm::InputTag("electronMVAVariableHelper", "kfhits")),
            cc.consumes<edm::ValueMap<float>>(edm::InputTag("electronMVAVariableHelper", "kfchi2")),
            cc.consumes<edm::ValueMap<float>>(edm::InputTag("electronMVAVariableHelper", "convVtxFitProb")),
            cc.consumes<double>(edm::InputTag("fixedGridRhoFastjetAll"))
        })
{}

template<>
const std::vector<float> MVAVariableHelper<reco::GsfElectron>::getAuxVariables(
        edm::Ptr<reco::GsfElectron> const& particlePtr, const edm::Event& iEvent) const
{
    return std::vector<float> {
        getVariableFromValueMapToken(particlePtr, tokens_[0], iEvent),
        getVariableFromValueMapToken(particlePtr, tokens_[1], iEvent),
        getVariableFromValueMapToken(particlePtr, tokens_[2], iEvent),
        getVariableFromDoubleToken(tokens_[3], iEvent)
    };
}

template<>
MVAVariableIndexMap<reco::GsfElectron>::MVAVariableIndexMap()
    : indexMap_({
            {"electronMVAVariableHelper:kfhits"        , 0},
            {"electronMVAVariableHelper:kfchi2"        , 1},
            {"electronMVAVariableHelper:convVtxFitProb", 2},
            {"fixedGridRhoFastjetAll"                  , 3}
        })
{}

/////////////
// Specializations for the Photons
/////////////

#include "DataFormats/EgammaCandidates/interface/Photon.h"

template<>
MVAVariableHelper<reco::Photon>::MVAVariableHelper(edm::ConsumesCollector && cc)
    : tokens_({
            cc.consumes<edm::ValueMap<float>>(edm::InputTag("photonIDValueMapProducer", "phoPhotonIsolation")),
            cc.consumes<edm::ValueMap<float>>(edm::InputTag("photonIDValueMapProducer", "phoChargedIsolation")),
            cc.consumes<edm::ValueMap<float>>(edm::InputTag("photonIDValueMapProducer", "phoWorstChargedIsolation")),
            cc.consumes<edm::ValueMap<float>>(edm::InputTag("photonIDValueMapProducer", "phoWorstChargedIsolationConeVeto")),
            cc.consumes<edm::ValueMap<float>>(edm::InputTag("photonIDValueMapProducer", "phoWorstChargedIsolationConeVetoPVConstr")),
            cc.consumes<edm::ValueMap<float>>(edm::InputTag("egmPhotonIsolation", "gamma-DR030-")),
            cc.consumes<edm::ValueMap<float>>(edm::InputTag("egmPhotonIsolation", "h+-DR030-")),
            cc.consumes<double>(edm::InputTag("fixedGridRhoFastjetAll")),
            cc.consumes<double>(edm::InputTag("fixedGridRhoAll"))
        })
{}

template<>
const std::vector<float> MVAVariableHelper<reco::Photon>::getAuxVariables(
        edm::Ptr<reco::Photon> const& particlePtr, const edm::Event& iEvent) const
{
    return std::vector<float> {
        getVariableFromValueMapToken(particlePtr, tokens_[0], iEvent),
        getVariableFromValueMapToken(particlePtr, tokens_[1], iEvent),
        getVariableFromValueMapToken(particlePtr, tokens_[2], iEvent),
        getVariableFromValueMapToken(particlePtr, tokens_[3], iEvent),
        getVariableFromValueMapToken(particlePtr, tokens_[4], iEvent),
        getVariableFromValueMapToken(particlePtr, tokens_[5], iEvent),
        getVariableFromValueMapToken(particlePtr, tokens_[6], iEvent),
        getVariableFromDoubleToken(tokens_[7], iEvent),
        getVariableFromDoubleToken(tokens_[8], iEvent)
    };
}

template<>
MVAVariableIndexMap<reco::Photon>::MVAVariableIndexMap()
    : indexMap_({
            {"photonIDValueMapProducer:phoPhotonIsolation"                      , 0},
            {"photonIDValueMapProducer:phoChargedIsolation"                     , 1},
            {"photonIDValueMapProducer:phoWorstChargedIsolation"                , 2},
            {"photonIDValueMapProducer:phoWorstChargedIsolationConeVeto"        , 3},
            {"photonIDValueMapProducer:phoWorstChargedIsolationConeVetoPVConstr", 4},
            {"egmPhotonIsolation:gamma-DR030-"                                  , 5},
            {"egmPhotonIsolation:h+-DR030-"                                     , 6},
            {"fixedGridRhoFastjetAll"                                           , 7},
            {"fixedGridRhoAll"                                                  , 8}
        })
{}
