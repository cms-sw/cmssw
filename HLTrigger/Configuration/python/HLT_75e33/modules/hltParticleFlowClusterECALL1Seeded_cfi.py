import FWCore.ParameterSet.Config as cms

hltParticleFlowClusterECALL1Seeded = cms.EDProducer("CorrectedECALPFClusterProducer",
    energyCorrector = cms.PSet(
        applyCrackCorrections = cms.bool(False),
        applyMVACorrections = cms.bool(True),
        autoDetectBunchSpacing = cms.bool(True),
        bunchSpacing = cms.int32(25),
        ebSrFlagLabel = cms.InputTag("hltEcalDigis"),
        eeSrFlagLabel = cms.InputTag("hltEcalDigis"),
        maxPtForMVAEvaluation = cms.double(300.0),
        recHitsEBLabel = cms.InputTag("hltRechitInRegionsECAL","EcalRecHitsEB"),
        recHitsEELabel = cms.InputTag("hltRechitInRegionsECAL","EcalRecHitsEE"),
        setEnergyUncertainty = cms.bool(False),
        srfAwareCorrection = cms.bool(True)
    ),
    inputECAL = cms.InputTag("hltParticleFlowClusterECALUncorrectedL1Seeded"),
    inputPS = cms.InputTag("hltParticleFlowClusterPSL1Seeded"),
    mightGet = cms.optional.untracked.vstring,
    minimumPSEnergy = cms.double(0)
)
