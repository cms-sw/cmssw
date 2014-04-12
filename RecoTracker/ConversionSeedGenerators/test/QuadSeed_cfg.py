import FWCore.ParameterSet.Config as cms
#knuenz, Nov 2012

process = cms.Process('quad')

process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.GlobalTag.globaltag = 'START53_V7A::All'
from FWCore.MessageService.MessageLogger_cfi import *
MessageLogger.cerr.FwkReport.reportEvery = 100

#process.load("Workspace.TrackProducerFromSeed.TrackProducerFromSeed_cfi") #UserCode/WAdam/Workspace/TrackProducerFromSeed
#process.load("RecoEgamma.Examples.simpleConvertedPhotonAnalyzer_cfi")

from RecoTracker.ConversionSeedGenerators.QuadReReco_cff import quadrereco

process=quadrereco(process)

process.source = cms.Source(
    "PoolSource",
    fileNames = cms.untracked.vstring('file:root://eoscms//eos/cms/store/mc/Summer12_DR53X/MinBias_TuneZ2star_8TeV-pythia6/GEN-SIM-RECODEBUG/DEBUG_PU_S10_START53_V7A-v1/0000/04B96853-D1E0-E111-86D0-002618943845.root')
    #fileNames = cms.untracked.vstring('file:root://eoscms//eos/cms/store/relval/CMSSW_6_1_0_pre6-START61_V5/RelValMinBias/GEN-SIM-RECO/v1/00000/642ECE84-E433-E211-ADB0-003048FFD720.root')
    )

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(100)
    )

process.options = cms.untracked.PSet(
    wantSummary = cms.untracked.bool(False),
    )

process.out = cms.OutputModule("PoolOutputModule",
                               fileName = cms.untracked.string('quad_ReReco.root'),
                               SelectEvents = cms.untracked.PSet(SelectEvents = cms.vstring('QuadFinalPath')
                                                                 )
                               )

#process.Tracer = cms.Service("Tracer")
#process.Timing = cms.Service("Timing")
 
process.QuadFinalPath = cms.Path(
                                 #process.convertedPhotonAnalyzer*
                                 process.quadrereco
                                 #*process.tracksFromQuadSeeds
                                 )

s = process.dumpPython()
f = open("tmp.py","w")
f.write(s)
f.close()

process.e = cms.EndPath(process.out)

process.schedule = cms.Schedule( process.QuadFinalPath, process.e )

