# A simple cfg to prescale input events. Used to test the production system. 

import FWCore.ParameterSet.Config as cms

process = cms.Process("AOD")



from CMGTools.Production.datasetToSource import *
process.source = datasetToSource(
   'cbern',
   '/DoubleMu/Run2012B-PromptReco-v1/AOD/V5_Test',
   '.*root'
   )


process.options   = cms.untracked.PSet( wantSummary = cms.untracked.bool(False))
#WARNING!
process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(100) )


process.pre = cms.EDFilter("Prescaler",
                           prescaleFactor = cms.int32(2),
                           prescaleOffset = cms.int32(0)
                           )

process.p = cms.Path(
    process.pre
    )

process.out = cms.OutputModule(
    "PoolOutputModule",
    outputCommands = cms.untracked.vstring( 'keep *'),
    fileName = cms.untracked.string('prescale.root'),
    SelectEvents = cms.untracked.PSet( SelectEvents = cms.vstring('p') )
    )


process.endpath = cms.EndPath(
    process.out
    )


process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.MessageLogger.cerr.FwkReport.reportEvery = 100


