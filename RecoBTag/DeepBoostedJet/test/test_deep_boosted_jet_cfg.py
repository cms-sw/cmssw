
import FWCore.ParameterSet.Config as cms
from PhysicsTools.PatAlgos.tools.helpers import getPatAlgosToolsTask

from FWCore.ParameterSet.VarParsing import VarParsing
options = VarParsing('analysis')
options.inputFiles = '/store/mc/RunIIFall17MiniAOD/ZprimeToWWToWlepWhad_narrow_M-3000_TuneCP5_13TeV-madgraph/MINIAODSIM/94X_mc2017_realistic_v10-v1/20000/3E25D208-8205-E811-8858-3417EBE64426.root'
options.maxEvents = -1
options.parseArguments()

process = cms.Process("PATtest")

## MessageLogger
process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.MessageLogger.cerr.FwkReport.reportEvery = 100


## Options and Output Report
process.options = cms.untracked.PSet( wantSummary = cms.untracked.bool(True) )

## Source
process.source = cms.Source("PoolSource",
    fileNames=cms.untracked.vstring(options.inputFiles)
)
## Maximal Number of Events
process.maxEvents = cms.untracked.PSet(input=cms.untracked.int32(options.maxEvents))

## Geometry and Detector Conditions (needed for a few patTuple production steps)
process.load("Configuration.Geometry.GeometryRecoDB_cff")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:run2_mc')
process.load("Configuration.StandardSequences.MagneticField_cff")

## Output Module Configuration (expects a path 'p')
from PhysicsTools.PatAlgos.patEventContent_cff import patEventContentNoCleaning
process.out = cms.OutputModule("PoolOutputModule",
                               fileName = cms.untracked.string('patTuple.root'),
                               ## save only events passing the full path
                               #SelectEvents = cms.untracked.PSet( SelectEvents = cms.vstring('p') ),
                               ## save PAT output; you need a '*' to unpack the list of commands
                               ## 'patEventContent'
                               outputCommands = cms.untracked.vstring('drop *', *patEventContentNoCleaning )
                               )

patAlgosToolsTask = getPatAlgosToolsTask(process)
process.outpath = cms.EndPath(process.out, patAlgosToolsTask)

## and add them to the event content
from PhysicsTools.PatAlgos.tools.jetTools import updateJetCollection

updateJetCollection(
   process,
   jetSource = cms.InputTag('slimmedJetsAK8'),
   pvSource = cms.InputTag('offlineSlimmedPrimaryVertices'),
   svSource = cms.InputTag('slimmedSecondaryVertices'),
   rParam = 0.8,
   jetCorrections = ('AK8PFPuppi', cms.vstring(['L2Relative', 'L3Absolute']), 'None'),
   btagDiscriminators = [
           # DeepBoostedJet (Nominal)
          'pfDeepBoostedJetTags:probTbcq',
          'pfDeepBoostedJetTags:probTbqq',
          'pfDeepBoostedJetTags:probTbc',
          'pfDeepBoostedJetTags:probTbq',
          'pfDeepBoostedJetTags:probWcq',
          'pfDeepBoostedJetTags:probWqq',
          'pfDeepBoostedJetTags:probZbb',
          'pfDeepBoostedJetTags:probZcc',
          'pfDeepBoostedJetTags:probZqq',
          'pfDeepBoostedJetTags:probHbb',
          'pfDeepBoostedJetTags:probHcc',
          'pfDeepBoostedJetTags:probHqqqq',
          'pfDeepBoostedJetTags:probQCDbb',
          'pfDeepBoostedJetTags:probQCDcc',
          'pfDeepBoostedJetTags:probQCDb',
          'pfDeepBoostedJetTags:probQCDc',
          'pfDeepBoostedJetTags:probQCDothers',
           # meta taggers
          'pfDeepBoostedDiscriminatorsJetTags:TvsQCD',
          'pfDeepBoostedDiscriminatorsJetTags:WvsQCD',

           # DeepBoostedJet (mass decorrelated)
          'pfMassDecorrelatedDeepBoostedJetTags:probTbcq',
          'pfMassDecorrelatedDeepBoostedJetTags:probTbqq',
          'pfMassDecorrelatedDeepBoostedJetTags:probTbc',
          'pfMassDecorrelatedDeepBoostedJetTags:probTbq',
          'pfMassDecorrelatedDeepBoostedJetTags:probWcq',
          'pfMassDecorrelatedDeepBoostedJetTags:probWqq',
          'pfMassDecorrelatedDeepBoostedJetTags:probZbb',
          'pfMassDecorrelatedDeepBoostedJetTags:probZcc',
          'pfMassDecorrelatedDeepBoostedJetTags:probZqq',
          'pfMassDecorrelatedDeepBoostedJetTags:probHbb',
          'pfMassDecorrelatedDeepBoostedJetTags:probHcc',
          'pfMassDecorrelatedDeepBoostedJetTags:probHqqqq',
          'pfMassDecorrelatedDeepBoostedJetTags:probQCDbb',
          'pfMassDecorrelatedDeepBoostedJetTags:probQCDcc',
          'pfMassDecorrelatedDeepBoostedJetTags:probQCDb',
          'pfMassDecorrelatedDeepBoostedJetTags:probQCDc',
          'pfMassDecorrelatedDeepBoostedJetTags:probQCDothers',
           # meta taggers
          'pfMassDecorrelatedDeepBoostedDiscriminatorsJetTags:TvsQCD',
          'pfMassDecorrelatedDeepBoostedDiscriminatorsJetTags:WvsQCD',
          'pfMassDecorrelatedDeepBoostedDiscriminatorsJetTags:bbvsQCD',
          'pfMassDecorrelatedDeepBoostedDiscriminatorsJetTags:ZHbbvsQCD',
          'pfMassDecorrelatedDeepBoostedDiscriminatorsJetTags:ccvsQCD',
          'pfMassDecorrelatedDeepBoostedDiscriminatorsJetTags:ZHccvsQCD',
          'pfMassDecorrelatedDeepBoostedDiscriminatorsJetTags:bbvscc',
          'pfMassDecorrelatedDeepBoostedDiscriminatorsJetTags:ZHbbvsZHcc',

      ]
   )

from Configuration.EventContent.EventContent_cff import MINIAODSIMEventContent
process.out.outputCommands.append('keep *_slimmedJetsAK8*_*_*')
process.out.outputCommands.append('keep *_offlineSlimmedPrimaryVertices*_*_*')
process.out.outputCommands.append('keep *_slimmedSecondaryVertices*_*_*')
process.out.outputCommands.append('keep *_selectedPatJets*_*_*')
process.out.outputCommands.append('keep *_selectedUpdatedPatJets*_*_*')
process.out.outputCommands.append('keep *_updatedPatJets*_*_*')

process.out.fileName = 'test_deep_boosted_jet_MINIAODSIM.root'

#                                         ##
#   process.options.wantSummary = False   ##  (to suppress the long output at the end of the job)
# process.add_(cms.Service("InitRootHandlers", DebugLevel =cms.untracked.int32(3)))
