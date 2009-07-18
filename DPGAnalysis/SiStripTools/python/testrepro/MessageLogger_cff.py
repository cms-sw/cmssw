import FWCore.ParameterSet.Config as cms

from FWCore.MessageLogger.MessageLogger_cfi import MessageLogger

MessageLogger.infos.placeholder = cms.untracked.bool(False)
MessageLogger.infos.threshold = cms.untracked.string("INFO")
MessageLogger.infos.default = cms.untracked.PSet(
    limit = cms.untracked.int32(10000000)
    )
MessageLogger.infos.FwkReport = cms.untracked.PSet(
    reportEvery = cms.untracked.int32(5000)
    )
MessageLogger.cerr.threshold = cms.untracked.string("WARNING")

# following lines required to avoid too many warning messages
MessageLogger.categories.append("NoSimpleCluster")
MessageLogger.categories.append("TrackProducer")
MessageLogger.categories.append("SiStripMonitorTrack")
MessageLogger.categories.append("SiStripRecHitConverter")
#
MessageLogger.infos.NoSimpleCluster = cms.untracked.PSet(
    limit = cms.untracked.int32(0)
    )
MessageLogger.cerr.NoSimpleCluster = cms.untracked.PSet(
    limit = cms.untracked.int32(0)
    )
MessageLogger.infos.TrackProducer = cms.untracked.PSet(
    limit = cms.untracked.int32(0)
    )
MessageLogger.infos.SiStripMonitorTrack= cms.untracked.PSet(
    limit = cms.untracked.int32(0)
    )
MessageLogger.infos.SiStripRecHitConverter= cms.untracked.PSet(
    limit = cms.untracked.int32(0)
    )
#
MessageLogger.suppressInfo.append("offlineBeamSpot")

