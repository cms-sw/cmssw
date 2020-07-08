import FWCore.ParameterSet.Config as cms
from PhysicsTools.PatAlgos.tools.helpers import getPatAlgosToolsTask

from FWCore.ParameterSet.VarParsing import VarParsing
options = VarParsing('analysis')
options.inputFiles = '/store/mc/RunIIFall17MiniAODv2/TTToHadronic_TuneCP5_13TeV-powheg-pythia8/MINIAODSIM/PU2017_12Apr2018_94X_mc2017_realistic_v14-v1/90000/DCFE3F5F-AE42-E811-B6DB-008CFAF72A64.root'
options.maxEvents = 1000
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
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:phase1_2017_realistic')
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
from RecoBTag.ONNXRuntime.pfParticleNet_cff import _pfParticleNetJetTagsAll as pfParticleNetJetTagsAll

updateJetCollection(
   process,
   jetSource = cms.InputTag('slimmedJetsAK8'),
   pvSource = cms.InputTag('offlineSlimmedPrimaryVertices'),
   svSource = cms.InputTag('slimmedSecondaryVertices'),
   rParam = 0.8,
   jetCorrections = ('AK8PFPuppi', cms.vstring(['L2Relative', 'L3Absolute']), 'None'),
   btagDiscriminators = pfParticleNetJetTagsAll
   )

from Configuration.EventContent.EventContent_cff import MINIAODSIMEventContent
process.out.outputCommands.append('keep *_slimmedJetsAK8*_*_*')
process.out.outputCommands.append('keep *_offlineSlimmedPrimaryVertices*_*_*')
process.out.outputCommands.append('keep *_slimmedSecondaryVertices*_*_*')
process.out.outputCommands.append('keep *_selectedPatJets*_*_*')
process.out.outputCommands.append('keep *_selectedUpdatedPatJets*_*_*')
process.out.outputCommands.append('keep *_updatedPatJets*_*_*')

process.out.fileName = 'test_particle_net_MINIAODSIM.root'
