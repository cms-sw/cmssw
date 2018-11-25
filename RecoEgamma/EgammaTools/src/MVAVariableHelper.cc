#include "RecoEgamma/EgammaTools/interface/MVAVariableHelper.h"

/////////////
// Specializations for the Electrons
/////////////

#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "DataFormats/EgammaCandidates/interface/Conversion.h"
#include "RecoEgamma/EgammaTools/interface/ConversionTools.h"
#include "RecoEgamma/ElectronIdentification/interface/ElectronMVAEstimatorRun2.h"

template<>
MVAVariableHelper<reco::GsfElectron>::MVAVariableHelper(edm::ConsumesCollector && cc)
    : tokens_({
            cc.consumes<reco::ConversionCollection>(edm::InputTag("allConversions")),
            cc.consumes<reco::ConversionCollection>(edm::InputTag("reducedEgamma:reducedConversions")),
            cc.consumes<reco::BeamSpot>(edm::InputTag("offlineBeamSpot")),
            cc.consumes<double>(edm::InputTag("fixedGridRhoFastjetAll"))
        })
{}

template<>
const std::vector<float> MVAVariableHelper<reco::GsfElectron>::getAuxVariables(
        edm::Ptr<reco::GsfElectron> const& particlePtr, const edm::Event& iEvent) const
{
    edm::Handle<reco::ConversionCollection> conversionsHandle;
    edm::Handle<reco::BeamSpot> beamSpotHandle;
    edm::Handle<double> rhoHandle;

    iEvent.getByToken(tokens_[0], conversionsHandle);
    if( !conversionsHandle.isValid() ) {
      iEvent.getByToken(tokens_[1], conversionsHandle);
      if( !conversionsHandle.isValid() )
        throw cms::Exception(" Collection not found: ")
            << " failed to find a standard AOD or miniAOD conversions collection " << std::endl;
    }

    iEvent.getByToken(tokens_[2], beamSpotHandle);
    iEvent.getByToken(tokens_[3], rhoHandle);

    return ElectronMVAEstimatorRun2::getExtraVars(*particlePtr,
                                                  conversionsHandle.product(),
                                                  beamSpotHandle.product(),
                                                  *rhoHandle);
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
