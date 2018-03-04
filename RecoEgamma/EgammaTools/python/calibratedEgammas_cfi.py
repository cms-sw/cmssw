import FWCore.ParameterSet.Config as cms

calibratedEgammaSettings = cms.PSet(minEtToCalibrate = cms.double(5.0),
                                    semiDeterministic = cms.bool(True),
                                    correctionFile = cms.string("EgammaAnalysis/ElectronTools/data/ScalesSmearings/Run2017_17Nov2017_v1_ele_unc"),
                                    recHitCollectionEB = cms.InputTag('reducedEcalRecHitsEB'),
                                    recHitCollectionEE = cms.InputTag('reducedEcalRecHitsEE'),
                                    produceCalibratedObjs = cms.bool(True)
                                    )

ecalTrkCombinationRegression = cms.PSet(
    ecalTrkRegressionConfig = cms.PSet(
        rangeMin = cms.double(-1.),
        rangeMax = cms.double(3.0),
        lowEtHighEtBoundary = cms.double(50.),
        forceHighEnergyTrainingIfSaturated = cms.bool(True),
        ebLowEtForestName = cms.string('electron_eb_ECALTRK_lowpt'),
        ebHighEtForestName = cms.string('electron_eb_ECALTRK'),
        eeLowEtForestName = cms.string('electron_ee_ECALTRK_lowpt'),
        eeHighEtForestName = cms.string('electron_ee_ECALTRK')
        ),
    ecalTrkRegressionUncertConfig = cms.PSet(
        rangeMin = cms.double(0.0002),
        rangeMax = cms.double(0.5),
        lowEtHighEtBoundary = cms.double(50.),  
        forceHighEnergyTrainingIfSaturated = cms.bool(True),
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
                                     # input collections
                                     src = cms.InputTag('gedGsfElectrons'),
                                     epCombConfig = ecalTrkCombinationRegression,
                                     )

calibratedPatElectrons = cms.EDProducer("CalibratedPatElectronProducer",
                                        calibratedEgammaSettings,
                                        # input collections
                                        src = cms.InputTag('slimmedElectrons'), 
                                        epCombConfig = ecalTrkCombinationRegression,
                                       )

calibratedPhotons = cms.EDProducer("CalibratedPhotonProducer",
                                   calibratedEgammaSettings,
                                   # input collections
                                   src = cms.InputTag('gedPhotons'),    
                                  )
calibratedPatPhotons = cms.EDProducer("CalibratedPatPhotonProducer",
                                      calibratedEgammaSettings,
                                      # input collections
                                      src = cms.InputTag('slimmedPhotons'),
                                      )
