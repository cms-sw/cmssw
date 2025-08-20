import FWCore.ParameterSet.Config as cms

hltOnlineBeamSpot = cms.EDProducer("BeamSpotOnlineProducer",
    changeToCMSCoordinates = cms.bool(False),
    gtEvmLabel = cms.InputTag(""),
    maxRadius = cms.double(2.0),
    maxZ = cms.double(40.0),
    setSigmaZ = cms.double(0.0),
    useBSOnlineRecords = cms.bool(True),
    timeThreshold = cms.int32(48),
    sigmaZThreshold = cms.double( 2.0 ),
    sigmaXYThreshold = cms.double( 4.0 )
)
