
import FWCore.ParameterSet.Config as cms
from PhysicsTools.PatAlgos.tools.helpers import getPatAlgosToolsTask

process = cms.Process("PATtest")

## MessageLogger
process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.MessageLogger.cerr.FwkReport.reportEvery = 1000


## Options and Output Report
process.options = cms.untracked.PSet( wantSummary = cms.untracked.bool(True) )

## Source
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring()
)
## Maximal Number of Events
process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(-1) )

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
   jetCorrections = ('AK8PFchs', cms.vstring(['L2Relative', 'L3Absolute']), 'None'),
   btagDiscriminators = [
      'pfBoostedDoubleSecondaryVertexAK8BJetTags',
      'pfDeepDoubleBvLJetTags:probQCD', 
      'pfDeepDoubleBvLJetTags:probHbb',
      'pfDeepDoubleCvLJetTags:probQCD',
      'pfDeepDoubleCvLJetTags:probHcc',
      'pfDeepDoubleCvBJetTags:probHbb',
      'pfDeepDoubleCvBJetTags:probHcc',
      'pfMassIndependentDeepDoubleBvLJetTags:probQCD', 
      'pfMassIndependentDeepDoubleBvLJetTags:probHbb',
      'pfMassIndependentDeepDoubleCvLJetTags:probQCD',
      'pfMassIndependentDeepDoubleCvLJetTags:probHcc',
      'pfMassIndependentDeepDoubleCvBJetTags:probHbb',
      'pfMassIndependentDeepDoubleCvBJetTags:probHcc',
      ]
   )

from PhysicsTools.PatAlgos.patInputFiles_cff import filesRelValTTbarPileUpMINIAODSIM

process.source.fileNames = filesRelValTTbarPileUpMINIAODSIM
process.source.fileNames = cms.untracked.vstring(
'/store/relval/CMSSW_10_3_0_pre2/RelValTTbar_13/MINIAODSIM/PU25ns_103X_upgrade2018_realistic_v2-v1/20000/85820ACA-657B-BC44-AC74-AACD6D54B348.root'
#'/store/mc/RunIIFall17MiniAOD/GluGluHToBB_M125_13TeV_powheg_pythia8/MINIAODSIM/94X_mc2017_realistic_v10-v1/20000/C8932584-5006-E811-9840-141877410512.root',
#'/store/mc/RunIIFall17MiniAODv2/GluGluHToCC_M125_13TeV_powheg_pythia8/MINIAODSIM/PU2017_12Apr2018_94X_mc2017_realistic_v14-v2/30000/72164088-CB67-E811-9D0D-008CFA197AC4.root',
#'/store/mc/RunIIFall17MiniAOD/QCD_HT700to1000_TuneCP5_13TeV-madgraph-pythia8/MINIAODSIM/94X_mc2017_realistic_v10-v1/20000/C0F304A4-23FA-E711-942E-E0071B6CAD20.root'
)
process.maxEvents.input = 1000 

from Configuration.EventContent.EventContent_cff import MINIAODSIMEventContent
process.out.outputCommands.append('keep *_slimmedJetsAK8*_*_*')
process.out.outputCommands.append('keep *_offlineSlimmedPrimaryVertices*_*_*')
process.out.outputCommands.append('keep *_slimmedSecondaryVertices*_*_*')
process.out.outputCommands.append('keep *_selectedPatJets*_*_*')
process.out.outputCommands.append('keep *_selectedUpdatedPatJets*_*_*')
process.out.outputCommands.append('keep *_pfBoostedDoubleSVAK8TagInfos*_*_*')
process.out.outputCommands.append('keep *_pfDeepDoubleXTagInfos*_*_*')
process.out.outputCommands.append('keep *_updatedPatJets*_*_*')

process.out.fileName = 'output_test_DDX.root'

#                                         ##
#   process.options.wantSummary = False   ##  (to suppress the long output at the end of the job)
# process.add_(cms.Service("InitRootHandlers", DebugLevel =cms.untracked.int32(3)))
