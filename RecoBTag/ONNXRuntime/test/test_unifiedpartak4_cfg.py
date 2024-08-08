
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
process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(1000))

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
                               outputCommands = cms.untracked.vstring('drop *', *patEventContentNoCleaning )
                               )

patAlgosToolsTask = getPatAlgosToolsTask(process)
process.outpath = cms.EndPath(process.out, patAlgosToolsTask)

## and add them to the event content
from PhysicsTools.PatAlgos.tools.jetTools import updateJetCollection

updateJetCollection(
   process,
   jetSource = cms.InputTag('slimmedJetsPuppi'),
   pvSource = cms.InputTag('offlineSlimmedPrimaryVertices'),
   svSource = cms.InputTag('slimmedSecondaryVertices'),
   jetCorrections = ('AK4PFchs', cms.vstring(['L1FastJet', 'L2Relative', 'L3Absolute']), 'None'),
   btagDiscriminators = [
      'pfDeepFlavourJetTags:probb',
      'pfDeepFlavourJetTags:probbb',
      'pfDeepFlavourJetTags:problepb',
      'pfDeepFlavourJetTags:probc',
      'pfDeepFlavourJetTags:probuds',
      'pfDeepFlavourJetTags:probg',
      'pfUnifiedParticleTransformerAK4JetTags:probb',
      'pfUnifiedParticleTransformerAK4JetTags:probbb',
      'pfUnifiedParticleTransformerAK4JetTags:problepb',
      'pfUnifiedParticleTransformerAK4JetTags:probc',
      'pfUnifiedParticleTransformerAK4JetTags:probs',
      'pfUnifiedParticleTransformerAK4JetTags:probu',
      'pfUnifiedParticleTransformerAK4JetTags:probd',
      'pfUnifiedParticleTransformerAK4JetTags:probg',
      'pfParticleTransformerAK4JetTags:probb',
      'pfParticleTransformerAK4JetTags:probbb',
      'pfParticleTransformerAK4JetTags:problepb',
      'pfParticleTransformerAK4JetTags:probc',
      'pfParticleTransformerAK4JetTags:probuds',
      'pfParticleTransformerAK4JetTags:probg',
      'pfParticleNetFromMiniAODAK4PuppiCentralJetTags:probb',
      'pfParticleNetFromMiniAODAK4PuppiCentralJetTags:probc',
      'pfParticleNetFromMiniAODAK4PuppiCentralJetTags:probuds',
      'pfParticleNetFromMiniAODAK4PuppiCentralJetTags:probg',
      ]
   )

from PhysicsTools.PatAlgos.patInputFiles_cff import filesRelValTTbarPileUpMINIAODSIM

process.source.fileNames = filesRelValTTbarPileUpMINIAODSIM
process.source.fileNames = cms.untracked.vstring('/store/mc/Run3Summer23BPixMiniAODv4/TTtoLNu2Q_TuneCP5_13p6TeV_powheg-pythia8/MINIAODSIM/130X_mcRun3_2023_realistic_postBPix_v2-v3/2520000/00488681-4f49-4bdc-89e6-198da9e42a17.root')

process.maxEvents.input = 1000

from Configuration.EventContent.EventContent_cff import MINIAODSIMEventContent
process.out.outputCommands = MINIAODSIMEventContent.outputCommands
process.out.outputCommands.append('keep *_selectedUpdatedPatJets*_*_*')
process.out.outputCommands.append('keep *_pfDeepFlavourTagInfos*_*_*')
process.out.outputCommands.append('keep *_pfDeepFlavourJetTags*_*_*')
process.out.outputCommands.append('keep *_updatedPatJets*_*_*')

process.out.fileName = 'test_deep_unifiedpartak4_MINIAODSIM.root'
