
import FWCore.ParameterSet.Config as cms
from PhysicsTools.PatAlgos.tools.helpers import getPatAlgosToolsTask

process = cms.Process("PATtest")

## MessageLogger
process.load("FWCore.MessageLogger.MessageLogger_cfi")

## Options and Output Report
process.options = cms.untracked.PSet( wantSummary = cms.untracked.bool(True) )

## Source
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring()
)
## Maximal Number of Events
process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(100) )

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
   jetSource = cms.InputTag('slimmedJets'),
   pvSource = cms.InputTag('offlineSlimmedPrimaryVertices'),
   svSource = cms.InputTag('slimmedSecondaryVertices'),
   jetCorrections = ('AK4PFchs', cms.vstring(['L1FastJet', 'L2Relative', 'L3Absolute']), 'None'),
   btagDiscriminators = [
      'pfCombinedSecondaryVertexV2BJetTags',
      'pfDeepCSVJetTags:probudsg', 
      'pfDeepCSVJetTags:probb', 
      'pfDeepCSVJetTags:probc', 
      'pfDeepCSVJetTags:probbb', 
      'pfDeepFlavourJetTags:probb',
      'pfDeepFlavourJetTags:probbb',
      'pfDeepFlavourJetTags:problepb',
      'pfDeepFlavourJetTags:probc',
      'pfDeepFlavourJetTags:probuds',
      'pfDeepFlavourJetTags:probg',
      'pfDeepVertexJetTags:probb',
      'pfDeepCombinedJetTags:probb',

      ]
   )

from PhysicsTools.PatAlgos.patInputFiles_cff import filesRelValTTbarPileUpMINIAODSIM

process.source.fileNames = filesRelValTTbarPileUpMINIAODSIM
process.source.fileNames = cms.untracked.vstring('/store/mc/RunIIFall17MiniAODv2/TTToHadronic_TuneCP5_13TeV-powheg-pythia8/MINIAODSIM/PU2017_12Apr2018_new_pmx_94X_mc2017_realistic_v14-v2/60000/FCC2AFA9-4BBB-E811-B35F-0CC47AFB7D48.root')

process.maxEvents.input = 10

from Configuration.EventContent.EventContent_cff import MINIAODSIMEventContent
process.out.outputCommands = MINIAODSIMEventContent.outputCommands
process.out.outputCommands.append('keep *_selectedUpdatedPatJets*_*_*')
process.out.outputCommands.append('keep *_pfDeepCSVTagInfos*_*_*')
process.out.outputCommands.append('keep *_pfDeepFlavourTagInfos*_*_*')
process.out.outputCommands.append('keep *_pfDeepFlavourJetTags*_*_*')
process.out.outputCommands.append('keep *_updatedPatJets*_*_*')

process.out.fileName = 'test_deep_vertexcomb_MINIAODSIM.root'

#                                         ##
#   process.options.wantSummary = False   ##  (to suppress the long output at the end of the job)
# process.add_(cms.Service("InitRootHandlers", DebugLevel =cms.untracked.int32(3)))
