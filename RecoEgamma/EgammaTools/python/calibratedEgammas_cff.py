import FWCore.ParameterSet.Config as cms

calibratedEgammaSettings = cms.PSet(minEtToCalibrate = cms.double(5.0),
                                    semiDeterministic = cms.bool(True),
                                    correctionFile = cms.string("EgammaAnalysis/ElectronTools/data/ScalesSmearings/Run2017_17Nov2017_v1_ele_unc"),
                                    recHitCollectionEB = cms.InputTag('reducedEcalRecHitsEB'),
                                    recHitCollectionEE = cms.InputTag('reducedEcalRecHitsEE'),
                                    produceCalibratedObjs = cms.bool(True)
                                    )
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

calibratedElectrons = cms.EDProducer("CalibratedElectronProducer",
                                     calibratedEgammaSettings,                                   
                                     epCombConfig = ecalTrkCombinationRegression,
                                     src = cms.InputTag('gedGsfElectrons'),
                                     )

calibratedPatElectrons = cms.EDProducer("CalibratedPatElectronProducer",
                                        calibratedEgammaPatSettings,
                                        epCombConfig = ecalTrkCombinationRegression,
                                        src = cms.InputTag('slimmedElectrons'), 
                                       )

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
