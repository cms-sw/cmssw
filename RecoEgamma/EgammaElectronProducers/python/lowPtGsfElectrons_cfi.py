import FWCore.ParameterSet.Config as cms

from RecoEgamma.EgammaTools.regressionModifier_cfi import regressionModifier106XUL

lowPtRegressionModifier = regressionModifier106XUL.clone(
    modifierName = 'EGRegressionModifierV3',
    rhoTag = 'fixedGridRhoFastjetAll',
    eleRegs = dict(
        ecalOnlyMean = dict(
            lowEtHighEtBoundary = 20.,
            ebLowEtForestName = ":lowPtElectron_eb_ecalOnly_05To50_mean",
            ebHighEtForestName = ":lowPtElectron_eb_ecalOnly_05To50_mean",
            eeLowEtForestName = ":lowPtElectron_ee_ecalOnly_05To50_mean",
            eeHighEtForestName = ":lowPtElectron_ee_ecalOnly_05To50_mean",
            ),
        ecalOnlySigma = dict(
            lowEtHighEtBoundary = 20.,
            ebLowEtForestName = ":lowPtElectron_eb_ecalOnly_05To50_sigma",
            ebHighEtForestName = ":lowPtElectron_eb_ecalOnly_05To50_sigma",
            eeLowEtForestName = ":lowPtElectron_ee_ecalOnly_05To50_sigma",
            eeHighEtForestName = ":lowPtElectron_ee_ecalOnly_05To50_sigma",
            ),
        epComb = dict(
            ecalTrkRegressionConfig = dict(
                lowEtHighEtBoundary = 20.,
                ebLowEtForestName = ":lowPtElectron_eb_ecalTrk_05To50_mean",
                ebHighEtForestName = ":lowPtElectron_eb_ecalTrk_05To50_mean",
                eeLowEtForestName = ":lowPtElectron_ee_ecalTrk_05To50_mean",
                eeHighEtForestName = ":lowPtElectron_ee_ecalTrk_05To50_mean",
                ),
            ecalTrkRegressionUncertConfig = dict(
                lowEtHighEtBoundary = 20.,
                ebLowEtForestName = ":lowPtElectron_eb_ecalTrk_05To50_sigma",
                ebHighEtForestName = ":lowPtElectron_eb_ecalTrk_05To50_sigma",
                eeLowEtForestName = ":lowPtElectron_ee_ecalTrk_05To50_sigma",
                eeHighEtForestName = ":lowPtElectron_ee_ecalTrk_05To50_sigma",
                ),
        )
    ),
)

from RecoEgamma.EgammaTools.regressionModifier_cfi import regressionModifier103XLowPtPho
_lowPtRegressionModifierUPC = regressionModifier103XLowPtPho.clone(
    eleRegs = dict(
        ecalOnlyMean = dict(
            ebLowEtForestName = ":lowPtElectron_eb_ecalOnly_1To20_0p2To2_mean",
            ebHighEtForestName = ":lowPtElectron_eb_ecalOnly_1To20_0p2To2_mean",
            eeLowEtForestName = ":lowPtElectron_ee_ecalOnly_1To20_0p2To2_mean",
            eeHighEtForestName = ":lowPtElectron_ee_ecalOnly_1To20_0p2To2_mean",
            ),
        ecalOnlySigma = dict(
            ebLowEtForestName = ":lowPtElectron_eb_ecalOnly_1To20_0p0002To0p5_sigma",
            ebHighEtForestName = ":lowPtElectron_eb_ecalOnly_1To20_0p0002To0p5_sigma",
            eeLowEtForestName = ":lowPtElectron_ee_ecalOnly_1To20_0p0002To0p5_sigma",
            eeHighEtForestName = ":lowPtElectron_ee_ecalOnly_1To20_0p0002To0p5_sigma",
            ),
        epComb = dict(
            ecalTrkRegressionConfig = dict(
                ebLowEtForestName = ":lowPtElectron_eb_ecalTrk_1To20_0p2To2_mean",
                ebHighEtForestName = ":lowPtElectron_eb_ecalTrk_1To20_0p2To2_mean",
                eeLowEtForestName = ":lowPtElectron_ee_ecalTrk_1To20_0p2To2_mean",
                eeHighEtForestName = ":lowPtElectron_ee_ecalTrk_1To20_0p2To2_mean",
                ),
            ecalTrkRegressionUncertConfig = dict(
                ebLowEtForestName = ":lowPtElectron_eb_ecalTrk_1To20_0p0002To0p5_sigma",
                ebHighEtForestName = ":lowPtElectron_eb_ecalTrk_1To20_0p0002To0p5_sigma",
                eeLowEtForestName = ":lowPtElectron_ee_ecalTrk_1To20_0p0002To0p5_sigma",
                eeHighEtForestName = ":lowPtElectron_ee_ecalTrk_1To20_0p0002To0p5_sigma",
                ),
        )
    ),
)
from Configuration.Eras.Modifier_run3_upc_2023_cff import run3_upc_2023
from Configuration.Eras.Modifier_run3_upc_2025_cff import run3_upc_2025
from Configuration.ProcessModifiers.egamma_lowPt_exclusive_cff import egamma_lowPt_exclusive
(egamma_lowPt_exclusive & (run3_upc_2023 | run3_upc_2025)).toReplaceWith(lowPtRegressionModifier,_lowPtRegressionModifierUPC)

from RecoEgamma.EgammaElectronProducers.lowPtGsfElectronFinalizer_cfi import lowPtGsfElectronFinalizer
lowPtGsfElectrons = lowPtGsfElectronFinalizer.clone(
    previousGsfElectronsTag = "lowPtGsfElectronsPreRegression",
    regressionConfig = lowPtRegressionModifier,
)

from Configuration.ProcessModifiers.run2_miniAOD_UL_cff import run2_miniAOD_UL
run2_miniAOD_UL.toModify(lowPtGsfElectrons, previousGsfElectronsTag = "lowPtGsfElectrons::@skipCurrentProcess")
