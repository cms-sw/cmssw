import FWCore.ParameterSet.Config as cms

lowPtGsfLinks = cms.EDProducer( 'LowPtGSFToPackedCandidateLinker',
    PFCandidates = cms.InputTag("particleFlow"),
    packedCandidates = cms.InputTag("packedPFCandidates"),
    lostTracks = cms.InputTag('lostTracks'),
    tracks = cms.InputTag("generalTracks"),
    gsfPreID = cms.InputTag("lowPtGsfElectronSeeds"),
    gsfTracks = cms.InputTag("lowPtGsfEleGsfTracks"),
    )
