import FWCore.ParameterSet.Config as cms

particleFlowClusterECAL = cms.EDProducer('CorrectedECALPFClusterProducer',
  minimumPSEnergy = cms.double(0),
  inputPS = cms.InputTag('particleFlowClusterPS'),
  energyCorrector = cms.PSet(
    applyCrackCorrections = cms.bool(False),
    applyMVACorrections = cms.bool(False),
    srfAwareCorrection = cms.bool(False),
    setEnergyUncertainty = cms.bool(False),
    autoDetectBunchSpacing = cms.bool(True),
    bunchSpacing = cms.int32(25),
    maxPtForMVAEvaluation = cms.double(-99),
    algoName = cms.string('PFClusterEMEnergyCorrector'),
    recHitsEBLabel = cms.InputTag('ecalRecHit', 'EcalRecHitsEB'),
    recHitsEELabel = cms.InputTag('ecalRecHit', 'EcalRecHitsEE'),
    verticesLabel = cms.InputTag('offlinePrimaryVertices'),
    ebSrFlagLabel = cms.InputTag('ecalDigis'),
    eeSrFlagLabel = cms.InputTag('ecalDigis')
  ),
  inputECAL = cms.InputTag('particleFlowClusterECALUncorrected'),
  mightGet = cms.optional.untracked.vstring
)
