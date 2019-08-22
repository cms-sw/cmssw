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
            cc.consumes<double>(edm::InputTag("fixedGridRhoFastjetAll"))
        })
{}

template<>
const std::vector<float> MVAVariableHelper<reco::GsfElectron>::getAuxVariables(
        edm::Ptr<reco::GsfElectron> const& particlePtr, const edm::Event& iEvent) const
{
    edm::Handle<double> rhoHandle;
    iEvent.getByToken(tokens_[0], rhoHandle);
    
    //design made much more sense when it wasnt just rho...
    return ElectronMVAEstimatorRun2::getExtraVars(*rhoHandle);
}

template<>
MVAVariableIndexMap<reco::GsfElectron>::MVAVariableIndexMap()
    : indexMap_({
            {"fixedGridRhoFastjetAll"                  , 0}
        })
{}

/////////////
// Specializations for the Photons
/////////////

#include "DataFormats/EgammaCandidates/interface/Photon.h"

template<>
MVAVariableHelper<reco::Photon>::MVAVariableHelper(edm::ConsumesCollector && cc)
    : tokens_({
            cc.consumes<double>(edm::InputTag("fixedGridRhoFastjetAll")),
            cc.consumes<double>(edm::InputTag("fixedGridRhoAll"))
        })
{}

template<>
const std::vector<float> MVAVariableHelper<reco::Photon>::getAuxVariables(
        edm::Ptr<reco::Photon> const& particlePtr, const edm::Event& iEvent) const
{
    return std::vector<float> {
        getVariableFromDoubleToken(tokens_[0], iEvent),
        getVariableFromDoubleToken(tokens_[1], iEvent)
    };
}

template<>
MVAVariableIndexMap<reco::Photon>::MVAVariableIndexMap()
    : indexMap_({
            {"fixedGridRhoFastjetAll"                                           , 0},
            {"fixedGridRhoAll"                                                  , 1}
        })
{}
