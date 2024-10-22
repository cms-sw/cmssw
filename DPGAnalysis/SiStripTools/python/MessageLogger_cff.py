import FWCore.ParameterSet.Config as cms

from FWCore.MessageService.MessageLogger_cfi import MessageLogger

#----------------------------------------------------------------

MessageLogger.cout.enable = True
MessageLogger.cout.threshold = cms.untracked.string("INFO")
MessageLogger.cout.default = cms.untracked.PSet(
    limit = cms.untracked.int32(10000000)
    )
MessageLogger.cout.FwkReport = cms.untracked.PSet(
    reportEvery = cms.untracked.int32(10000)
    )

MessageLogger.cerr.enable = True
MessageLogger.cerr.threshold = cms.untracked.string("WARNING")
MessageLogger.cerr.default = cms.untracked.PSet(
    limit = cms.untracked.int32(10000000)
    )
MessageLogger.cerr.FwkReport = cms.untracked.PSet(
    reportEvery = cms.untracked.int32(100000)
    )

#----Remove too verbose PrimaryVertexProducer

MessageLogger.suppressInfo.append("pixelVerticesAdaptive")
MessageLogger.suppressInfo.append("pixelVerticesAdaptiveNoBS")

#----Remove too verbose BeamSpotOnlineProducer

MessageLogger.suppressInfo.append("testBeamSpot")
MessageLogger.suppressInfo.append("onlineBeamSpot")
MessageLogger.suppressWarning.append("testBeamSpot")
MessageLogger.suppressWarning.append("onlineBeamSpot")

#----Remove too verbose TrackRefitter

MessageLogger.suppressInfo.append("newTracksFromV0")
MessageLogger.suppressInfo.append("newTracksFromOtobV0")


#------------------------------------------------------------------


