import FWCore.ParameterSet.Config as cms
import subprocess
#import FWCore.Utilities.FileUtils as FileUtils

process = cms.Process("DQMPlots")

process.load("FWCore.MessageService.MessageLogger_cfi")
process.MessageLogger.cerr.threshold = 'INFO'
process.MessageLogger.categories.append('DQMPlots')
process.MessageLogger.cerr.INFO = cms.untracked.PSet(
	limit = cms.untracked.int32(-1)
)

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(-1) )
process.load('Configuration.StandardSequences.Services_cff')
process.load('Configuration.EventContent.EventContent_cff')
process.load('Configuration.Geometry.GeometryExtended2016_cff')
process.load('Configuration.Geometry.GeometryExtended2016Reco_cff')



readFiles = cms.untracked.vstring()
process.source = cms.Source("PoolSource",
        fileNames = readFiles,
)
in_dir_name = './'
readFiles.extend( cms.untracked.vstring('file:'+in_dir_name+'test_cppf_emulator.root') )


process.load('L1Trigger.L1TMuonCPPF.cppf_dqm_cfi')
process.TFileService = cms.Service("TFileService",
	fileName = cms.string("DQM_CPPF.root")
)
process.p = cms.Path(process.DQM_CPPF)


