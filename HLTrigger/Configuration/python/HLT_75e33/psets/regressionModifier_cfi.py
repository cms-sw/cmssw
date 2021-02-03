import FWCore.ParameterSet.Config as cms

regressionModifier = cms.PSet(
    eleRegs = cms.PSet(
        ecalOnlyMean = cms.PSet(
            ebHighEtForestName = cms.string('electron_eb_ECALonly'),
            ebLowEtForestName = cms.string('electron_eb_ecalOnly_1To300_0p2To2_mean'),
            eeHighEtForestName = cms.string('electron_ee_ECALonly'),
            eeLowEtForestName = cms.string('electron_ee_ecalOnly_1To300_0p2To2_mean'),
            forceHighEnergyTrainingIfSaturated = cms.bool(True),
            lowEtHighEtBoundary = cms.double(999999.0),
            rangeMaxHighEt = cms.double(3.0),
            rangeMaxLowEt = cms.double(2.0),
            rangeMinHighEt = cms.double(-1.0),
            rangeMinLowEt = cms.double(0.2)
        ),
        ecalOnlySigma = cms.PSet(
            ebHighEtForestName = cms.string('electron_eb_ECALonly_var'),
            ebLowEtForestName = cms.string('electron_eb_ecalOnly_1To300_0p0002To0p5_sigma'),
            eeHighEtForestName = cms.string('electron_ee_ECALonly_var'),
            eeLowEtForestName = cms.string('electron_ee_ecalOnly_1To300_0p0002To0p5_sigma'),
            forceHighEnergyTrainingIfSaturated = cms.bool(True),
            lowEtHighEtBoundary = cms.double(999999.0),
            rangeMaxHighEt = cms.double(0.5),
            rangeMaxLowEt = cms.double(0.5),
            rangeMinHighEt = cms.double(0.0002),
            rangeMinLowEt = cms.double(0.0002)
        ),
        epComb = cms.PSet(
            ecalTrkRegressionConfig = cms.PSet(
                ebHighEtForestName = cms.string('electron_eb_ecalTrk_1To300_0p2To2_mean'),
                ebLowEtForestName = cms.string('electron_eb_ecalTrk_1To300_0p2To2_mean'),
                eeHighEtForestName = cms.string('electron_ee_ecalTrk_1To300_0p2To2_mean'),
                eeLowEtForestName = cms.string('electron_ee_ecalTrk_1To300_0p2To2_mean'),
                forceHighEnergyTrainingIfSaturated = cms.bool(False),
                lowEtHighEtBoundary = cms.double(999999.0),
                rangeMaxHighEt = cms.double(2.0),
                rangeMaxLowEt = cms.double(2.0),
                rangeMinHighEt = cms.double(0.2),
                rangeMinLowEt = cms.double(0.2)
            ),
            ecalTrkRegressionUncertConfig = cms.PSet(
                ebHighEtForestName = cms.string('electron_eb_ecalTrk_1To300_0p0002To0p5_sigma'),
                ebLowEtForestName = cms.string('electron_eb_ecalTrk_1To300_0p0002To0p5_sigma'),
                eeHighEtForestName = cms.string('electron_ee_ecalTrk_1To300_0p0002To0p5_sigma'),
                eeLowEtForestName = cms.string('electron_ee_ecalTrk_1To300_0p0002To0p5_sigma'),
                forceHighEnergyTrainingIfSaturated = cms.bool(False),
                lowEtHighEtBoundary = cms.double(999999.0),
                rangeMaxHighEt = cms.double(0.5),
                rangeMaxLowEt = cms.double(0.5),
                rangeMinHighEt = cms.double(0.0002),
                rangeMinLowEt = cms.double(0.0002)
            ),
            maxEPDiffInSigmaForComb = cms.double(15.0),
            maxEcalEnergyForComb = cms.double(200.0),
            maxRelTrkMomErrForComb = cms.double(10.0),
            minEOverPForComb = cms.double(0.025)
        )
    ),
    maxRawEnergyForLowPtEBSigma = cms.double(-1),
    maxRawEnergyForLowPtEESigma = cms.double(1200.0),
    modifierName = cms.string('EGRegressionModifierV3'),
    phoRegs = cms.PSet(
        ecalOnlyMean = cms.PSet(
            ebHighEtForestName = cms.string('photon_eb_ECALonly'),
            ebLowEtForestName = cms.string('photon_eb_ecalOnly_5To300_0p2To2_mean'),
            eeHighEtForestName = cms.string('photon_ee_ECALonly'),
            eeLowEtForestName = cms.string('photon_ee_ecalOnly_5To300_0p2To2_mean'),
            forceHighEnergyTrainingIfSaturated = cms.bool(True),
            lowEtHighEtBoundary = cms.double(999999.0),
            rangeMaxHighEt = cms.double(3.0),
            rangeMaxLowEt = cms.double(2.0),
            rangeMinHighEt = cms.double(-1.0),
            rangeMinLowEt = cms.double(0.2)
        ),
        ecalOnlySigma = cms.PSet(
            ebHighEtForestName = cms.string('photon_eb_ECALonly_var'),
            ebLowEtForestName = cms.string('photon_eb_ecalOnly_5To300_0p0002To0p5_sigma'),
            eeHighEtForestName = cms.string('photon_ee_ECALonly_var'),
            eeLowEtForestName = cms.string('photon_ee_ecalOnly_5To300_0p0002To0p5_sigma'),
            forceHighEnergyTrainingIfSaturated = cms.bool(True),
            lowEtHighEtBoundary = cms.double(999999.0),
            rangeMaxHighEt = cms.double(0.5),
            rangeMaxLowEt = cms.double(0.5),
            rangeMinHighEt = cms.double(0.0002),
            rangeMinLowEt = cms.double(0.0002)
        )
    ),
    rhoTag = cms.InputTag("fixedGridRhoFastjetAllTmp"),
    useClosestToCentreSeedCrysDef = cms.bool(False)
)