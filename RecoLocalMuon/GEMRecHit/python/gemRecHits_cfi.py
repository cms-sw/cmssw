import FWCore.ParameterSet.Config as cms

gemRecHits = cms.EDProducer("GEMRecHitProducer",
    recAlgoConfig = cms.PSet(),
    recAlgo = cms.string('GEMRecHitStandardAlgo'),
    gemDigiLabel = cms.InputTag("muonGEMDigis"),
    applyMasking = cms.bool(False),
    maskSource = cms.string(''),
    #maskvecfile = cms.FileInPath('RecoLocalMuon/GEMRecHit/data/GEMMaskVec.dat'),
    deadSource = cms.string(''),
    #deadvecfile = cms.FileInPath('RecoLocalMuon/GEMRecHit/data/GEMDeadVec.dat')
)
