import FWCore.ParameterSet.Config as cms

hltParticleFlowClusterECALMuonSeeded = cms.EDProducer("CorrectedECALPFClusterProducer",
    energyCorrector = cms.PSet(
        applyCrackCorrections = cms.bool(False),
        applyMVACorrections = cms.bool(True),
        autoDetectBunchSpacing = cms.bool(True),
        bunchSpacing = cms.int32(25),
        ebSrFlagLabel = cms.InputTag("hltEcalDigis"),
        eeSrFlagLabel = cms.InputTag("hltEcalDigis"),
        maxPtForMVAEvaluation = cms.double(300.0),
        recHitsEBLabel = cms.InputTag("hltEcalRecHitMuonSeeded","EcalRecHitsEB"),
        recHitsEELabel = cms.InputTag("hltEcalRecHitMuonSeeded","EcalRecHitsEE"),
        setEnergyUncertainty = cms.bool(False),
        srfAwareCorrection = cms.bool(True)
    ),
    inputECAL = cms.InputTag("hltParticleFlowClusterECALUncorrectedMuonSeeded"),
    mightGet = cms.optional.untracked.vstring,
    minimumPSEnergy = cms.double(0),
    skipPS = cms.bool(True)
)
