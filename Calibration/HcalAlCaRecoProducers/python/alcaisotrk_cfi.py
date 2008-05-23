import FWCore.ParameterSet.Config as cms

# producer for alcaisotrk (HCAL isolated tracks)
from TrackingTools.TrackAssociator.default_cfi import *
IsoProd = cms.EDProducer("AlCaIsoTracksProducer",
    TrackAssociatorParameterBlock,
    hbheInput = cms.InputTag("hbhereco"),
    hoInput = cms.InputTag("horeco"),
    pCut = cms.untracked.double(2.0),
    ecalInputs = cms.VInputTag(cms.InputTag("ecalRecHit","EcalRecHitsEB"), cms.InputTag("ecalRecHit","EcalRecHitsEE")),
    histoFlag = cms.untracked.int32(0),
    ptCut = cms.untracked.double(1.5),
    inputTrackLabel = cms.untracked.string('generalTracks')
)


