import FWCore.ParameterSet.Config as cms

#### PF CLUSTER ECAL ####

#energy corrector for corrected cluster producer
_emEnergyCorrector = cms.PSet(
    algoName = cms.string("PFClusterEMEnergyCorrector"),
    applyCrackCorrections = cms.bool(False),
    applyMVACorrections = cms.bool(True),
    recHitsEBLabel = cms.InputTag('ecalRecHit', 'EcalRecHitsEB'),
    recHitsEELabel = cms.InputTag('ecalRecHit', 'EcalRecHitsEE'),
    autoDetectBunchSpacing = cms.bool(True),
)

particleFlowClusterECAL = cms.EDProducer(
    "CorrectedECALPFClusterProducer",
    inputECAL = cms.InputTag("particleFlowClusterECALUncorrected"),
    inputPS = cms.InputTag("particleFlowClusterPS"),
    minimumPSEnergy = cms.double(0.0),
    energyCorrector = _emEnergyCorrector
    )

