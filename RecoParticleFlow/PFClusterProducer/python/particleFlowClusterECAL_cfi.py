import FWCore.ParameterSet.Config as cms

#### PF CLUSTER ECAL ####

#energy corrector for corrected cluster producer
_emEnergyCorrector = cms.PSet(
    algoName = cms.string('PFClusterEMEnergyCorrector'),
    applyCrackCorrections = cms.bool(False),
    applyMVACorrections = cms.bool(True),
    maxPtForMVAEvaluation = cms.double(90.),
    recHitsEBLabel = cms.InputTag('ecalRecHit', 'EcalRecHitsEB'),
    recHitsEELabel = cms.InputTag('ecalRecHit', 'EcalRecHitsEE'),
    ebSrFlagLabel = cms.InputTag('ecalDigis'),
    eeSrFlagLabel = cms.InputTag('ecalDigis'),
    autoDetectBunchSpacing = cms.bool(True),
    srfAwareCorrection = cms.bool(False)
)

_emEnergyCorrector_2017 = cms.PSet(
    algoName = cms.string('PFClusterEMEnergyCorrector'),
    applyCrackCorrections = cms.bool(False),
    applyMVACorrections = cms.bool(True),
    maxPtForMVAEvaluation = cms.double(300.),
    recHitsEBLabel = cms.InputTag('ecalRecHit', 'EcalRecHitsEB'),
    recHitsEELabel = cms.InputTag('ecalRecHit', 'EcalRecHitsEE'),
    ebSrFlagLabel = cms.InputTag('ecalDigis'),
    eeSrFlagLabel = cms.InputTag('ecalDigis'),
    autoDetectBunchSpacing = cms.bool(True),
    srfAwareCorrection = cms.bool(True)
)

particleFlowClusterECAL = cms.EDProducer(
    'CorrectedECALPFClusterProducer',
    inputECAL = cms.InputTag('particleFlowClusterECALUncorrected'),
    inputPS = cms.InputTag('particleFlowClusterPS'),
    minimumPSEnergy = cms.double(0.0),
    energyCorrector = _emEnergyCorrector
    )


from Configuration.Eras.Modifier_run2_ECAL_2017_cff import run2_ECAL_2017
run2_ECAL_2017.toReplaceWith(_emEnergyCorrector, _emEnergyCorrector_2017)
