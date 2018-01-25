#!/usr/bin/env cmsRun
import FWCore.ParameterSet.Config as cms
import sys

process = cms.Process("LHE")

#find the path to the config file
path = "/".join(sys.argv[1].split('/')[0:-1])
if not path:
    path = '.'

process.source = cms.Source("LHESource",
	fileNames = cms.untracked.vstring('file:'+path+'/ttbar.lhe','file:'+path+'/ttbar_nonMergeable.lhe')
)

process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(-1))

process.configurationMetadata = cms.untracked.PSet(
	version = cms.untracked.string('alpha'),
	name = cms.untracked.string('LHEF input'),
	annotation = cms.untracked.string('ttbar')
)

process.load("FWCore.MessageService.MessageLogger_cfi")
#process.MessageLogger.cerr.threshold = 'INFO'
process.MessageLogger.cerr = cms.untracked.PSet( threshold = cms.untracked.string('INFO') )

process.LHE = cms.OutputModule("PoolOutputModule",
	dataset = cms.untracked.PSet(dataTier = cms.untracked.string('LHE')),
	fileName = cms.untracked.string('lhe_noMerge.root')
)

process.lhedump = cms.EDAnalyzer("DummyLHEAnalyzer",
                                 src = cms.InputTag("source"),
                                 dumpHeader = cms.untracked.bool(True)
                                 )

process.p = cms.Path(process.lhedump)
process.outpath = cms.EndPath(process.LHE)

