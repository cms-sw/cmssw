import FWCore.ParameterSet.Config as cms

from RecoEgamma.EgammaTools.regressionModifier_cfi import regressionModifier106XUL

lowPtRegressionModifier = regressionModifier106XUL.clone(
    modifierName = 'EGRegressionModifierV3',
    rhoTag = 'fixedGridRhoFastjetAllTmp',
    useClosestToCentreSeedCrysDef = False,
    maxRawEnergyForLowPtEBSigma = -1.,
    maxRawEnergyForLowPtEESigma = 1200.,
    eleRegs = dict(
        ecalOnlyMean = dict(
            rangeMinLowEt = 0.2,
            rangeMaxLowEt = 2.0,
            rangeMinHighEt = -1.,
            rangeMaxHighEt = 3.0,
            forceHighEnergyTrainingIfSaturated = True,
            lowEtHighEtBoundary = 20.,
            ebLowEtForestName = ":lowPtElectron_eb_ecalOnly_05To50_mean",
            ebHighEtForestName = ":lowPtElectron_eb_ecalOnly_05To50_mean",
            eeLowEtForestName = ":lowPtElectron_ee_ecalOnly_05To50_mean",
            eeHighEtForestName = ":lowPtElectron_ee_ecalOnly_05To50_mean",
            ),
        ecalOnlySigma = dict(
            rangeMinLowEt = 0.0002,
            rangeMaxLowEt = 0.5,
            rangeMinHighEt = 0.0002,
            rangeMaxHighEt = 0.5,
            forceHighEnergyTrainingIfSaturated = True,
            lowEtHighEtBoundary = 20.,
            ebLowEtForestName = ":lowPtElectron_eb_ecalOnly_05To50_sigma",
            ebHighEtForestName = ":lowPtElectron_eb_ecalOnly_05To50_sigma",
            eeLowEtForestName = ":lowPtElectron_ee_ecalOnly_05To50_sigma",
            eeHighEtForestName = ":lowPtElectron_ee_ecalOnly_05To50_sigma",
            ),
        epComb = dict(
            ecalTrkRegressionConfig = dict(
                rangeMinLowEt = 0.2,
                rangeMaxLowEt = 2.0,
                rangeMinHighEt = 0.2,
                rangeMaxHighEt = 2.0,
                lowEtHighEtBoundary = 20.,
                forceHighEnergyTrainingIfSaturated = False,
                ebLowEtForestName = ":lowPtElectron_eb_ecalTrk_05To50_mean",
                ebHighEtForestName = ":lowPtElectron_eb_ecalTrk_05To50_mean",
                eeLowEtForestName = ":lowPtElectron_ee_ecalTrk_05To50_mean",
                eeHighEtForestName = ":lowPtElectron_ee_ecalTrk_05To50_mean",
                ),
            ecalTrkRegressionUncertConfig = dict(
                rangeMinLowEt = 0.0002,
                rangeMaxLowEt = 0.5,
                rangeMinHighEt = 0.0002,
                rangeMaxHighEt = 0.5,
                lowEtHighEtBoundary = 20.,
                forceHighEnergyTrainingIfSaturated = False,
                ebLowEtForestName = ":lowPtElectron_eb_ecalTrk_05To50_sigma",
                ebHighEtForestName = ":lowPtElectron_eb_ecalTrk_05To50_sigma",
                eeLowEtForestName = ":lowPtElectron_ee_ecalTrk_05To50_sigma",
                eeHighEtForestName = ":lowPtElectron_ee_ecalTrk_05To50_sigma",
                ),
            maxEcalEnergyForComb = 200.,
            minEOverPForComb = 0.025,
            maxEPDiffInSigmaForComb = 15.,
            maxRelTrkMomErrForComb = 10.,
        )
    ),
)

lowPtGsfElectrons = cms.EDProducer("LowPtGsfElectronFinalizer",
                                   previousGsfElectronsTag = cms.InputTag("lowPtGsfElectronsPreRegression"),
                                   regressionConfig = lowPtRegressionModifier,
)
