import FWCore.ParameterSet.Config as cms

_correctionFile2016Legacy = "EgammaAnalysis/ElectronTools/data/ScalesSmearings/Legacy2016_07Aug2017_FineEtaR9_v3_ele_unc"
_correctionFile2017Nov17 = "EgammaAnalysis/ElectronTools/data/ScalesSmearings/Run2017_17Nov2017_v1_ele_unc"

calibratedEgammaSettings = cms.PSet(minEtToCalibrate = cms.double(5.0),
                                    semiDeterministic = cms.bool(True),
                                    correctionFile = cms.string(_correctionFile2017Nov17),
                                    recHitCollectionEB = cms.InputTag('reducedEcalRecHitsEB'),
                                    recHitCollectionEE = cms.InputTag('reducedEcalRecHitsEE'),
                                    produceCalibratedObjs = cms.bool(True)
                                    )
from Configuration.Eras.Modifier_run2_miniAOD_80XLegacy_cff import run2_miniAOD_80XLegacy
run2_miniAOD_80XLegacy.toModify(calibratedEgammaSettings,correctionFile = _correctionFile2016Legacy)

calibratedEgammaPatSettings = calibratedEgammaSettings.clone(
    recHitCollectionEB = cms.InputTag('reducedEgamma','reducedEBRecHits'),
    recHitCollectionEE = cms.InputTag('reducedEgamma','reducedEERecHits')
    )

ecalTrkCombinationRegression = cms.PSet(
    ecalTrkRegressionConfig = cms.PSet(
        rangeMin = cms.double(-1.),
        rangeMax = cms.double(3.0),
        lowEtHighEtBoundary = cms.double(50.),
        forceHighEnergyTrainingIfSaturated = cms.bool(False),
        ebLowEtForestName = cms.string('electron_eb_ECALTRK_lowpt'),
        ebHighEtForestName = cms.string('electron_eb_ECALTRK'),
        eeLowEtForestName = cms.string('electron_ee_ECALTRK_lowpt'),
        eeHighEtForestName = cms.string('electron_ee_ECALTRK')
        ),
    ecalTrkRegressionUncertConfig = cms.PSet(
        rangeMin = cms.double(0.0002),
        rangeMax = cms.double(0.5),
        lowEtHighEtBoundary = cms.double(50.),  
        forceHighEnergyTrainingIfSaturated = cms.bool(False),
        ebLowEtForestName = cms.string('electron_eb_ECALTRK_lowpt_var'),
        ebHighEtForestName = cms.string('electron_eb_ECALTRK_var'),
        eeLowEtForestName = cms.string('electron_ee_ECALTRK_lowpt_var'),
        eeHighEtForestName = cms.string('electron_ee_ECALTRK_var')
        ),
    maxEcalEnergyForComb=cms.double(200.),
    minEOverPForComb=cms.double(0.025),
    maxEPDiffInSigmaForComb=cms.double(15.),
    maxRelTrkMomErrForComb=cms.double(10.),
    
)

#a bug was found, the smear corrected ecal energy err was used in the E/p combination
#as the campaign has already started, we can not change the default nor the behaviour for the 94X Fall17 reminiAOD
#therefore we set useSmearCorrEcalEnergyErrInComb = cms.bool(True) when it should be set to false
#make no mistake, this is an incorrect configuration which will cause a scale shift at Et = 50 GeV
calibratedElectrons = cms.EDProducer("CalibratedElectronProducer",
                                     calibratedEgammaSettings,
                                     useSmearCorrEcalEnergyErrInComb = cms.bool(True),
                                     epCombConfig = ecalTrkCombinationRegression,
                                     src = cms.InputTag('gedGsfElectrons'),
                                     )

calibratedPatElectrons = cms.EDProducer("CalibratedPatElectronProducer",
                                        calibratedEgammaPatSettings,
                                        useSmearCorrEcalEnergyErrInComb = cms.bool(True),
                                        epCombConfig = ecalTrkCombinationRegression,
                                        src = cms.InputTag('slimmedElectrons'), 
                                       )
#the 80XLegacy miniAOD had not started, hence we can fix it for that
run2_miniAOD_80XLegacy.toModify(calibratedElectrons,useSmearCorrEcalEnergyErrInComb = False)
run2_miniAOD_80XLegacy.toModify(calibratedPatElectrons,useSmearCorrEcalEnergyErrInComb = False)



calibratedPhotons = cms.EDProducer("CalibratedPhotonProducer",
                                   calibratedEgammaSettings,
                                   src = cms.InputTag('gedPhotons'),    
                                  )
calibratedPatPhotons = cms.EDProducer("CalibratedPatPhotonProducer",
                                      calibratedEgammaPatSettings,
                                      src = cms.InputTag('slimmedPhotons'),
                                      )

def prefixName(prefix,name):
    return prefix+name[0].upper()+name[1:]



