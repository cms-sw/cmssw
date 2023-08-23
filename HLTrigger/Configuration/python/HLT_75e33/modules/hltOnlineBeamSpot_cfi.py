import FWCore.ParameterSet.Config as cms

hltOnlineBeamSpot = cms.EDProducer("BeamSpotOnlineProducer",
                                   useTransientRecord = cms.bool(True),
                                   changeToCMSCoordinates = cms.bool(False),
                                   gtEvmLabel = cms.InputTag(""),
                                   maxRadius = cms.double(2.0),
                                   maxZ = cms.double(40.0),
                                   setSigmaZ = cms.double(0.0))
