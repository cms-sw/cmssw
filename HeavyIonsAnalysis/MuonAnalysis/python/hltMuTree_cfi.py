import FWCore.ParameterSet.Config as cms

hltMuTree = cms.EDAnalyzer("HLTMuTree",
                           muons = cms.InputTag("unpackedMuons"),
                           vertices = cms.InputTag("offlineSlimmedPrimaryVertices"),
                           doReco = cms.untracked.bool(True),
                           doGen = cms.untracked.bool(True),
                           genparticle = cms.InputTag("packedGenParticles"),
                           simtrack = cms.InputTag("mergedtruth","MergedTrackTruth"),
)
