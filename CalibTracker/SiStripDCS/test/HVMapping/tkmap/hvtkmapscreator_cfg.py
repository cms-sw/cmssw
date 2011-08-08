import FWCore.ParameterSet.Config as cms
import sys

process = cms.Process("HVTKMapsCreator")

process.load("FWCore.MessageService.MessageLogger_cfi")

process.MessageLogger.infos.placeholder = cms.untracked.bool(False)
process.MessageLogger.infos.threshold = cms.untracked.string("INFO")
process.MessageLogger.infos.default = cms.untracked.PSet(
    limit = cms.untracked.int32(10000000)
    )
process.MessageLogger.infos.FwkReport = cms.untracked.PSet(
    reportEvery = cms.untracked.int32(10000)
    )
process.MessageLogger.cerr.threshold = cms.untracked.string("WARNING")


process.source = cms.Source("EmptySource",
                            firstRun = cms.untracked.uint32(77777),
                            numberEventsInRun = cms.untracked.uint32(1)
                            )

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(1) )

process.TkDetMap = cms.Service("TkDetMap")
process.load("DQMServices.Core.DQMStore_cfg")
process.SiStripDetInfoFileReader = cms.Service("SiStripDetInfoFileReader")

process.load("myTKAnalyses.PSTools.hvtkmapcreator_cfi")

process.hvtkmapcreator.hvReassChannelFile = cms.string(sys.argv[2])
process.p0 = cms.Path(process.hvtkmapcreator)
