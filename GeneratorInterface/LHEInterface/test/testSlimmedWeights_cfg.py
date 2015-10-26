#!/usr/bin/env cmsRun
import FWCore.ParameterSet.Config as cms

process = cms.Process("SlimmedWeights")

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring('file:/afs/cern.ch/user/g/govoni/work/public/2015-10-13_LHEweights/CMSSW_7_1_14/src/HIG-RunIIWinter15GenOnly-00001.root')
)

process.load("FWCore.MessageService.MessageLogger_cfi")
#process.MessageLogger.cerr.threshold = 'INFO'
process.MessageLogger.cout = cms.untracked.PSet( threshold = cms.untracked.string('INFO') )

process.addSlimmedWeights = cms.EDProducer("LHESlimmedWeightsProducer")

process.output = cms.OutputModule("PoolOutputModule",
	fileName = cms.untracked.string('testSlimmedWeights.root')
)

process.p = cms.Path(process.addSlimmedWeights)

process.outpath = cms.EndPath(process.output)
