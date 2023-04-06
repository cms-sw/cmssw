import FWCore.ParameterSet.Config as cms

hltOnlineBeamSpotESProducer = cms.ESProducer("OnlineBeamSpotESProducer",
    timeThreshold = cms.int32( 48 ),
    sigmaZThreshold = cms.double( 2.0 ),
    sigmaXYThreshold = cms.double( 4.0 )
)
