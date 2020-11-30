import FWCore.ParameterSet.Config as cms

from RecoEgamma.EgammaTools.regressionModifier_cfi import regressionModifier106XUL

lowPtRegressionModifier = cms.PSet(
    modifierName = cms.string('EGRegressionModifierV3'),
    rhoTag = cms.InputTag('fixedGridRhoFastjetAllTmp'),
    useClosestToCentreSeedCrysDef = cms.bool(False),
    maxRawEnergyForLowPtEBSigma = cms.double(-1),
    maxRawEnergyForLowPtEESigma = cms.double(1200.),
    eleRegs = cms.PSet(
        ecalOnlyMean = cms.PSet(
            rangeMinLowEt = cms.double(0.2),
            rangeMaxLowEt = cms.double(2.0),
            rangeMinHighEt = cms.double(-1.),
            rangeMaxHighEt = cms.double(3.0),
            forceHighEnergyTrainingIfSaturated = cms.bool(True),
            lowEtHighEtBoundary = cms.double(20.),
            ebLowEtForestName = cms.string("lowPtElectron_eb_ecalOnly_05To50_mean"),
            ebHighEtForestName = cms.string("lowPtElectron_eb_ecalOnly_05To50_mean"),
            eeLowEtForestName = cms.string("lowPtElectron_ee_ecalOnly_05To50_mean"),
            eeHighEtForestName = cms.string("lowPtElectron_ee_ecalOnly_05To50_mean"),
            ),
        ecalOnlySigma = cms.PSet(
            rangeMinLowEt = cms.double(0.0002),
            rangeMaxLowEt = cms.double(0.5),
            rangeMinHighEt = cms.double(0.0002),
            rangeMaxHighEt = cms.double(0.5),
            forceHighEnergyTrainingIfSaturated = cms.bool(True),
            lowEtHighEtBoundary = cms.double(20.),
            ebLowEtForestName = cms.string("lowPtElectron_eb_ecalOnly_05To50_sigma"),
            ebHighEtForestName = cms.string("lowPtElectron_eb_ecalOnly_05To50_sigma"),
            eeLowEtForestName = cms.string("lowPtElectron_ee_ecalOnly_05To50_sigma"),
            eeHighEtForestName = cms.string("lowPtElectron_ee_ecalOnly_05To50_sigma"),
            ),
        epComb = cms.PSet(
            ecalTrkRegressionConfig = cms.PSet(
                rangeMinLowEt = cms.double(0.2),
                rangeMaxLowEt = cms.double(2.0),
                rangeMinHighEt = cms.double(0.2),
                rangeMaxHighEt = cms.double(2.0),
                lowEtHighEtBoundary = cms.double(20.),
                forceHighEnergyTrainingIfSaturated = cms.bool(False),
                ebLowEtForestName = cms.string('lowPtElectron_eb_ecalTrk_05To50_mean'),
                ebHighEtForestName = cms.string('lowPtElectron_eb_ecalTrk_05To50_mean'),
                eeLowEtForestName = cms.string('lowPtElectron_ee_ecalTrk_05To50_mean'),
                eeHighEtForestName = cms.string('lowPtElectron_ee_ecalTrk_05To50_mean'),
                ),
            ecalTrkRegressionUncertConfig = cms.PSet(
                rangeMinLowEt = cms.double(0.0002),
                rangeMaxLowEt = cms.double(0.5),
                rangeMinHighEt = cms.double(0.0002),
                rangeMaxHighEt = cms.double(0.5),
                lowEtHighEtBoundary = cms.double(20.),
                forceHighEnergyTrainingIfSaturated = cms.bool(False),
                ebLowEtForestName = cms.string('lowPtElectron_eb_ecalTrk_05To50_sigma'),
                ebHighEtForestName = cms.string('lowPtElectron_eb_ecalTrk_05To50_sigma'),
                eeLowEtForestName = cms.string('lowPtElectron_ee_ecalTrk_05To50_sigma'),
                eeHighEtForestName = cms.string('lowPtElectron_ee_ecalTrk_05To50_sigma'),
                ),
            maxEcalEnergyForComb=cms.double(200.),
            minEOverPForComb=cms.double(0.025),
            maxEPDiffInSigmaForComb=cms.double(15.),
            maxRelTrkMomErrForComb=cms.double(10.),
            )
        ),
    # Let's just clone the photon configuration from the regular regression config, because the modifier expects
    # us to put something
    phoRegs = regressionModifier106XUL.phoRegs.clone()
)

lowPtGsfElectrons = cms.EDProducer("LowPtGsfElectronFinalizer",
                                   previousGsfElectronsTag = cms.InputTag("lowPtGsfElectronsPreRegression"),
                                   regressionConfig = lowPtRegressionModifier,
)
