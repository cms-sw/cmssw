import FWCore.ParameterSet.Config as cms

hltOnlineBeamSpotESProducer = cms.ESProducer("OnlineBeamSpotESProducer",
    timeThreshold = cms.int32( int(1e6) ),   # we do want to read the DB even if it's old
    sigmaZThreshold = cms.double( 2.0 ),
    sigmaXYThreshold = cms.double( 4.0 )
)
