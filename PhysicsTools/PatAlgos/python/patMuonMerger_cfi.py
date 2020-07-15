import FWCore.ParameterSet.Config as cms

mergedMuons = cms.EDProducer("PATMuonMerger",
                             muons     = cms.InputTag("slimmedMuons"), 
                             pfCandidates=cms.InputTag("packedPFCandidates"),
                             otherTracks = cms.InputTag("lostTracks"),
                             muonCut = cms.string("pt>15 && abs(eta)<2.4"),
                             pfCandidatesCut = cms.string("pt>15 && abs(eta)<2.4"),
                             lostTrackCut = cms.string("pt>15 && abs(eta)<2.4")
                         )
