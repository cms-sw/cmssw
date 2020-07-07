import FWCore.ParameterSet.Config as cms

jetPlusTrackAddonSeedProducer = cms.EDProducer(
    "JetPlusTrackAddonSeedProducer",
    srcCaloJets = cms.InputTag("ak4CaloJets"),
    srcTrackJets = cms.InputTag("ak4TrackJets"),
    srcPVs = cms.InputTag('offlinePrimaryVertices'),
    dRcone = cms.double(0.4),
    PFCandidates = cms.InputTag('packedPFCandidates'),
    towerMaker = cms.InputTag('towerMaker'),
    UsePAT = cms.bool(False)
)

