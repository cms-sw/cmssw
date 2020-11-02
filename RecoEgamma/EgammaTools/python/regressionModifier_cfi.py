import FWCore.ParameterSet.Config as cms

regressionModifier106XUL = cms.PSet(
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
            lowEtHighEtBoundary = cms.double(999999.),
            ebLowEtForestName = cms.ESInputTag("", "electron_eb_ecalOnly_1To300_0p2To2_mean"),
            ebHighEtForestName = cms.ESInputTag("", "electron_eb_ECALonly"),
            eeLowEtForestName = cms.ESInputTag("", "electron_ee_ecalOnly_1To300_0p2To2_mean"),
            eeHighEtForestName = cms.ESInputTag("", "electron_ee_ECALonly"),
            ),
        ecalOnlySigma = cms.PSet(
            rangeMinLowEt = cms.double(0.0002),
            rangeMaxLowEt = cms.double(0.5),
            rangeMinHighEt = cms.double(0.0002),
            rangeMaxHighEt = cms.double(0.5),
            forceHighEnergyTrainingIfSaturated = cms.bool(True),
            lowEtHighEtBoundary = cms.double(999999.),
            ebLowEtForestName = cms.ESInputTag("", "electron_eb_ecalOnly_1To300_0p0002To0p5_sigma"),
            ebHighEtForestName = cms.ESInputTag("", "electron_eb_ECALonly_var"),
            eeLowEtForestName = cms.ESInputTag("", "electron_ee_ecalOnly_1To300_0p0002To0p5_sigma"),
            eeHighEtForestName = cms.ESInputTag("", "electron_ee_ECALonly_var"),
            ),
        epComb = cms.PSet(
            ecalTrkRegressionConfig = cms.PSet(
                rangeMinLowEt = cms.double(0.2),
                rangeMaxLowEt = cms.double(2.0),
                rangeMinHighEt = cms.double(0.2),
                rangeMaxHighEt = cms.double(2.0),
                lowEtHighEtBoundary = cms.double(999999.),
                forceHighEnergyTrainingIfSaturated = cms.bool(False),
                ebLowEtForestName = cms.ESInputTag("", 'electron_eb_ecalTrk_1To300_0p2To2_mean'),
                ebHighEtForestName = cms.ESInputTag("", 'electron_eb_ecalTrk_1To300_0p2To2_mean'),
                eeLowEtForestName = cms.ESInputTag("", 'electron_ee_ecalTrk_1To300_0p2To2_mean'),
                eeHighEtForestName = cms.ESInputTag("", 'electron_ee_ecalTrk_1To300_0p2To2_mean'),
                ),
            ecalTrkRegressionUncertConfig = cms.PSet(
                rangeMinLowEt = cms.double(0.0002),
                rangeMaxLowEt = cms.double(0.5),
                rangeMinHighEt = cms.double(0.0002),
                rangeMaxHighEt = cms.double(0.5),
                lowEtHighEtBoundary = cms.double(999999.),  
                forceHighEnergyTrainingIfSaturated = cms.bool(False),
                ebLowEtForestName = cms.ESInputTag("", 'electron_eb_ecalTrk_1To300_0p0002To0p5_sigma'),
                ebHighEtForestName = cms.ESInputTag("", 'electron_eb_ecalTrk_1To300_0p0002To0p5_sigma'),
                eeLowEtForestName = cms.ESInputTag("", 'electron_ee_ecalTrk_1To300_0p0002To0p5_sigma'),
                eeHighEtForestName = cms.ESInputTag("", 'electron_ee_ecalTrk_1To300_0p0002To0p5_sigma'),
                ),
            maxEcalEnergyForComb=cms.double(200.),
            minEOverPForComb=cms.double(0.025),
            maxEPDiffInSigmaForComb=cms.double(15.),
            maxRelTrkMomErrForComb=cms.double(10.),                
            )
        ),
    phoRegs = cms.PSet(
        ecalOnlyMean = cms.PSet(
            rangeMinLowEt = cms.double(0.2),
            rangeMaxLowEt = cms.double(2.0),
            rangeMinHighEt = cms.double(-1.),
            rangeMaxHighEt = cms.double(3.0),
            forceHighEnergyTrainingIfSaturated = cms.bool(True),
            lowEtHighEtBoundary = cms.double(999999.),
            ebLowEtForestName = cms.ESInputTag("", "photon_eb_ecalOnly_5To300_0p2To2_mean"),
            ebHighEtForestName = cms.ESInputTag("", "photon_eb_ECALonly"),
            eeLowEtForestName = cms.ESInputTag("", "photon_ee_ecalOnly_5To300_0p2To2_mean"),
            eeHighEtForestName = cms.ESInputTag("", "photon_ee_ECALonly"),
            ),
        ecalOnlySigma = cms.PSet(
            rangeMinLowEt = cms.double(0.0002),
            rangeMaxLowEt = cms.double(0.5),
            rangeMinHighEt = cms.double(0.0002),
            rangeMaxHighEt = cms.double(0.5),
            forceHighEnergyTrainingIfSaturated = cms.bool(True),
            lowEtHighEtBoundary = cms.double(999999.),
            ebLowEtForestName = cms.ESInputTag("", "photon_eb_ecalOnly_5To300_0p0002To0p5_sigma"),
            ebHighEtForestName = cms.ESInputTag("", "photon_eb_ECALonly_var"),
            eeLowEtForestName = cms.ESInputTag("", "photon_ee_ecalOnly_5To300_0p0002To0p5_sigma"),
            eeHighEtForestName = cms.ESInputTag("", "photon_ee_ECALonly_var"),
        ),
    )
)

regressionModifier103XLowPtPho = cms.PSet(
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
            lowEtHighEtBoundary = cms.double(999999.),
            ebLowEtForestName = cms.ESInputTag("", "electron_eb_ecalOnly_1To20_0p2To2_mean"),
            ebHighEtForestName = cms.ESInputTag("", "electron_eb_ECALonly"),
            eeLowEtForestName = cms.ESInputTag("", "electron_ee_ecalOnly_1To20_0p2To2_mean"),
            eeHighEtForestName = cms.ESInputTag("", "electron_ee_ECALonly"),
            ),
        ecalOnlySigma = cms.PSet(
            rangeMinLowEt = cms.double(0.0002),
            rangeMaxLowEt = cms.double(0.5),
            rangeMinHighEt = cms.double(0.0002),
            rangeMaxHighEt = cms.double(0.5),
            forceHighEnergyTrainingIfSaturated = cms.bool(True),
            lowEtHighEtBoundary = cms.double(999999.),
            ebLowEtForestName = cms.ESInputTag("", "electron_eb_ecalOnly_1To20_0p0002To0p5_sigma"),
            ebHighEtForestName = cms.ESInputTag("", "electron_eb_ECALonly_var"),
            eeLowEtForestName = cms.ESInputTag("", "electron_ee_ecalOnly_1To20_0p0002To0p5_sigma"),
            eeHighEtForestName = cms.ESInputTag("", "electron_ee_ECALonly_var"),
            ),
        epComb = cms.PSet(
            ecalTrkRegressionConfig = cms.PSet(
                rangeMinLowEt = cms.double(0.2),
                rangeMaxLowEt = cms.double(2.0),
                rangeMinHighEt = cms.double(0.2),
                rangeMaxHighEt = cms.double(2.0),
                lowEtHighEtBoundary = cms.double(999999.),
                forceHighEnergyTrainingIfSaturated = cms.bool(False),
                ebLowEtForestName = cms.ESInputTag("", 'electron_eb_ecalTrk_1To20_0p2To2_mean'),
                ebHighEtForestName = cms.ESInputTag("", 'electron_eb_ecalTrk_1To20_0p2To2_mean'),
                eeLowEtForestName = cms.ESInputTag("", 'electron_ee_ecalTrk_1To20_0p2To2_mean'),
                eeHighEtForestName = cms.ESInputTag("", 'electron_ee_ecalTrk_1To20_0p2To2_mean'),
                ),
            ecalTrkRegressionUncertConfig = cms.PSet(
                rangeMinLowEt = cms.double(0.0002),
                rangeMaxLowEt = cms.double(0.5),
                rangeMinHighEt = cms.double(0.0002),
                rangeMaxHighEt = cms.double(0.5),
                lowEtHighEtBoundary = cms.double(999999.),  
                forceHighEnergyTrainingIfSaturated = cms.bool(False),
                ebLowEtForestName = cms.ESInputTag("", 'electron_eb_ecalTrk_1To20_0p0002To0p5_sigma'),
                ebHighEtForestName = cms.ESInputTag("", 'electron_eb_ecalTrk_1To20_0p0002To0p5_sigma'),
                eeLowEtForestName = cms.ESInputTag("", 'electron_ee_ecalTrk_1To20_0p0002To0p5_sigma'),
                eeHighEtForestName = cms.ESInputTag("", 'electron_ee_ecalTrk_1To20_0p0002To0p5_sigma'),
                ),
            maxEcalEnergyForComb=cms.double(200.),
            minEOverPForComb=cms.double(0.025),
            maxEPDiffInSigmaForComb=cms.double(15.),
            maxRelTrkMomErrForComb=cms.double(10.),                
            )
        ),
    phoRegs = cms.PSet(
        ecalOnlyMean = cms.PSet(
            rangeMinLowEt = cms.double(0.2),
            rangeMaxLowEt = cms.double(2.0),
            rangeMinHighEt = cms.double(-1.),
            rangeMaxHighEt = cms.double(3.0),
            forceHighEnergyTrainingIfSaturated = cms.bool(True),
            lowEtHighEtBoundary = cms.double(999999.),
            ebLowEtForestName = cms.ESInputTag("", "photon_eb_ecalOnly_1To20_0p2To2_mean"),
            ebHighEtForestName = cms.ESInputTag("", "photon_eb_ECALonly"),
            eeLowEtForestName = cms.ESInputTag("", "photon_ee_ecalOnly_1To20_0p2To2_mean"),
            eeHighEtForestName = cms.ESInputTag("", "photon_ee_ECALonly"),
            ),
        ecalOnlySigma = cms.PSet(
            rangeMinLowEt = cms.double(0.0002),
            rangeMaxLowEt = cms.double(0.5),
            rangeMinHighEt = cms.double(0.0002),
            rangeMaxHighEt = cms.double(0.5),
            forceHighEnergyTrainingIfSaturated = cms.bool(True),
            lowEtHighEtBoundary = cms.double(999999.),
            ebLowEtForestName = cms.ESInputTag("", "photon_eb_ecalOnly_1To20_0p0002To0p5_sigma"),
            ebHighEtForestName = cms.ESInputTag("", "photon_eb_ECALonly_var"),
            eeLowEtForestName = cms.ESInputTag("", "photon_ee_ecalOnly_1To20_0p0002To0p5_sigma"),
            eeHighEtForestName = cms.ESInputTag("", "photon_ee_ECALonly_var"),
        ),
    )
)

regressionModifier94X = \
    cms.PSet( modifierName    = cms.string('EGRegressionModifierV2'),  

              rhoCollection = cms.InputTag('fixedGridRhoFastjetAllTmp'),
              
              electron_config = cms.PSet( # EB, EE
                regressionKey  = cms.vstring('electron_eb_ECALonly_lowpt', 'electron_eb_ECALonly', 'electron_ee_ECALonly_lowpt', 'electron_ee_ECALonly',
                                             'electron_eb_ECALTRK_lowpt', 'electron_eb_ECALTRK', 'electron_ee_ECALTRK_lowpt', 'electron_ee_ECALTRK'),
                uncertaintyKey = cms.vstring('electron_eb_ECALonly_lowpt_var', 'electron_eb_ECALonly_var', 'electron_ee_ECALonly_lowpt_var', 'electron_ee_ECALonly_var',
                                             'electron_eb_ECALTRK_lowpt_var', 'electron_eb_ECALTRK_var', 'electron_ee_ECALTRK_lowpt_var', 'electron_ee_ECALTRK_var'),
                                          ),
              
              photon_config   = cms.PSet( # EB, EE
                regressionKey  = cms.vstring('photon_eb_ECALonly_lowpt', 'photon_eb_ECALonly', 'photon_ee_ECALonly_lowpt', 'photon_ee_ECALonly'),
                uncertaintyKey = cms.vstring('photon_eb_ECALonly_lowpt_var', 'photon_eb_ECALonly_var', 'photon_ee_ECALonly_lowpt_var', 'photon_ee_ECALonly_var'),
                                          ),

              lowEnergy_ECALonlyThr = cms.double(99999.),
              lowEnergy_ECALTRKThr = cms.double(50.),
              highEnergy_ECALTRKThr = cms.double(200.),
              eOverP_ECALTRKThr = cms.double(0.025),
              epDiffSig_ECALTRKThr = cms.double(15.),
              epSig_ECALTRKThr = cms.double(10.),
              forceHighEnergyEcalTrainingIfSaturated = cms.bool(True)

              )


regressionModifier80X = \
    cms.PSet( modifierName    = cms.string('EGRegressionModifierV1'),
              autoDetectBunchSpacing = cms.bool(True),
              applyExtraHighEnergyProtection = cms.bool(True),
              bunchSpacingTag = cms.InputTag("bunchSpacingProducer"),
              manualBunchSpacing = cms.int32(50),
              rhoCollection = cms.InputTag("fixedGridRhoFastjetAll"),
              vertexCollection = cms.InputTag("offlinePrimaryVertices"),
              electron_config = cms.PSet( # EB, EE                                                                                                                                                          
                                          regressionKey_25ns  = cms.vstring('gedelectron_EBCorrection_25ns', 'gedelectron_EECorrection_25ns'),
                                          uncertaintyKey_25ns = cms.vstring('gedelectron_EBUncertainty_25ns', 'gedelectron_EEUncertainty_25ns'),
                                          combinationKey_25ns   = cms.ESInputTag("", 'gedelectron_p4combination_25ns'),

                                          regressionKey_50ns  = cms.vstring('gedelectron_EBCorrection_50ns', 'gedelectron_EECorrection_50ns'),
                                          uncertaintyKey_50ns = cms.vstring('gedelectron_EBUncertainty_50ns', 'gedelectron_EEUncertainty_50ns'),
                                          combinationKey_50ns   = cms.ESInputTag("", 'gedelectron_p4combination_50ns'),
                                          ),

              photon_config   = cms.PSet( # EB, EE                                                                                                                                                          
                                          regressionKey_25ns  = cms.vstring('gedphoton_EBCorrection_25ns', 'gedphoton_EECorrection_25ns'),
                                          uncertaintyKey_25ns = cms.vstring('gedphoton_EBUncertainty_25ns', 'gedphoton_EEUncertainty_25ns'),

                                          regressionKey_50ns  = cms.vstring('gedphoton_EBCorrection_50ns', 'gedphoton_EECorrection_50ns'),
                                          uncertaintyKey_50ns = cms.vstring('gedphoton_EBUncertainty_50ns', 'gedphoton_EEUncertainty_50ns'),
                                          )
              )

#by default we use the regression inappropriate to the main purpose of this release
#life is simplier that way
regressionModifier = regressionModifier94X.clone()


from Configuration.Eras.Modifier_run2_egamma_2016_cff import run2_egamma_2016
from Configuration.Eras.Modifier_run2_egamma_2017_cff import run2_egamma_2017
from Configuration.Eras.Modifier_run2_egamma_2018_cff import run2_egamma_2018

(run2_egamma_2016 | run2_egamma_2017 | run2_egamma_2018).toReplaceWith(regressionModifier,regressionModifier106XUL)

from Configuration.ProcessModifiers.egamma_lowPt_exclusive_cff import egamma_lowPt_exclusive
egamma_lowPt_exclusive.toReplaceWith(regressionModifier,regressionModifier103XLowPtPho)
