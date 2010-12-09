import FWCore.ParameterSet.Config as cms

process = cms.Process("PAT")

## MessageLogger
process.load("FWCore.MessageLogger.MessageLogger_cfi")

## Options and Output Report
process.options   = cms.untracked.PSet( wantSummary = cms.untracked.bool(True) )

## Source
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
      '/store/relval/CMSSW_3_9_5/RelValTTbar/GEN-SIM-RECO/START39_V6-v1/0008/0AEEDFA4-88FA-DF11-B6FF-001A92811718.root'
    )
)
## Maximal Number of Events
process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(100) )

## Geometry and Detector Conditions (needed for a few patTuple production steps)
process.load("Configuration.StandardSequences.Geometry_cff")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
from Configuration.PyReleaseValidation.autoCond import autoCond
process.GlobalTag.globaltag = cms.string( autoCond[ 'startup' ] )
process.load("Configuration.StandardSequences.MagneticField_cff")
# DB access for JEC
import CondCore.DBCommon.CondDBCommon_cfi
process.dbJetCorrections = cms.ESSource( "PoolDBESSource"
, CondCore.DBCommon.CondDBCommon_cfi.CondDBCommon
, toGet   = cms.VPSet(
    cms.PSet(
      connect = cms.untracked.string( 'frontier://FrontierProd/CMS_COND_31X_PHYSICSTOOLS' )
    , record  = cms.string( 'JetCorrectionsRecord' )
    , tag     = cms.string( 'JetCorrectorParametersCollection_Spring10_V8_KT4PF' )
    , label   = cms.untracked.string( 'KT4PF' )
    )
  , cms.PSet(
      connect = cms.untracked.string( 'frontier://FrontierProd/CMS_COND_31X_PHYSICSTOOLS' )
    , record  = cms.string( 'JetCorrectionsRecord' )
    , tag     = cms.string( 'JetCorrectorParametersCollection_Spring10_V8_KT4Calo' )
    , label   = cms.untracked.string( 'KT4Calo' )
    )
  , cms.PSet(
      connect = cms.untracked.string( 'frontier://FrontierProd/CMS_COND_31X_PHYSICSTOOLS' )
    , record  = cms.string( 'JetCorrectionsRecord' )
    , tag     = cms.string( 'JetCorrectorParametersCollection_Spring10_V8_AK7Calo' )
    , label   = cms.untracked.string( 'AK7Calo' )
    )
  , cms.PSet(
      connect = cms.untracked.string( 'frontier://FrontierProd/CMS_COND_31X_PHYSICSTOOLS' )
    , record  = cms.string( 'JetCorrectionsRecord' )
    , tag     = cms.string( 'JetCorrectorParametersCollection_Spring10_V8_KT6PF' )
    , label   = cms.untracked.string( 'KT6PF' )
    )
  , cms.PSet(
      connect = cms.untracked.string( 'frontier://FrontierProd/CMS_COND_31X_PHYSICSTOOLS' )
    , record  = cms.string( 'JetCorrectionsRecord' )
    , tag     = cms.string( 'JetCorrectorParametersCollection_Spring10_V8_AK7PF' )
    , label   = cms.untracked.string( 'AK7PF' )
    )
  , cms.PSet(
      connect = cms.untracked.string( 'frontier://FrontierProd/CMS_COND_31X_PHYSICSTOOLS' )
    , record  = cms.string( 'JetCorrectionsRecord' )
    , tag     = cms.string( 'JetCorrectorParametersCollection_Spring10_V8_AK5TRK' )
    , label   = cms.untracked.string( 'AK5TRK' )
    )
  , cms.PSet(
      connect = cms.untracked.string( 'frontier://FrontierProd/CMS_COND_31X_PHYSICSTOOLS' )
    , record  = cms.string( 'JetCorrectionsRecord' )
    , tag     = cms.string( 'JetCorrectorParametersCollection_Spring10_V8_KT6Calo' )
    , label   = cms.untracked.string( 'KT6Calo' )
    )
  , cms.PSet(
      connect = cms.untracked.string( 'frontier://FrontierProd/CMS_COND_31X_PHYSICSTOOLS' )
    , record  = cms.string( 'JetCorrectionsRecord' )
    , tag     = cms.string( 'JetCorrectorParametersCollection_Spring10_V8_IC5PF' )
    , label   = cms.untracked.string( 'IC5PF' )
    )
  , cms.PSet(
      connect = cms.untracked.string( 'frontier://FrontierProd/CMS_COND_31X_PHYSICSTOOLS' )
    , record  = cms.string( 'JetCorrectionsRecord' )
    , tag     = cms.string( 'JetCorrectorParametersCollection_Spring10_V8_AK5Calo' )
    , label   = cms.untracked.string( 'AK5Calo' )
    )
  , cms.PSet(
      connect = cms.untracked.string( 'frontier://FrontierProd/CMS_COND_31X_PHYSICSTOOLS' )
    , record  = cms.string( 'JetCorrectionsRecord' )
    , tag     = cms.string( 'JetCorrectorParametersCollection_Spring10_V8_AK5PF' )
    , label   = cms.untracked.string( 'AK5PF' )
    )
  , cms.PSet(
      connect = cms.untracked.string( 'frontier://FrontierProd/CMS_COND_31X_PHYSICSTOOLS' )
    , record  = cms.string( 'JetCorrectionsRecord' )
    , tag     = cms.string( 'JetCorrectorParametersCollection_Summer10_V8_AK5JPT' )
    , label   = cms.untracked.string( 'AK5JPT' )
    )
  , cms.PSet(
      connect = cms.untracked.string( 'frontier://FrontierProd/CMS_COND_31X_PHYSICSTOOLS' )
    , record  = cms.string( 'JetCorrectionsRecord' )
    , tag     = cms.string( 'JetCorrectorParametersCollection_Spring10_V8_IC5Calo' )
    , label   = cms.untracked.string( 'IC5Calo' )
    )
  )
)
process.preferJetCorrections = cms.ESPrefer( "PoolDBESSource", "dbJetCorrections" )

## Standard PAT Configuration File
process.load("PhysicsTools.PatAlgos.patSequences_cff")

## Output Module Configuration (expects a path 'p')
from PhysicsTools.PatAlgos.patEventContent_cff import patEventContent
process.out = cms.OutputModule("PoolOutputModule",
                               fileName = cms.untracked.string('patTuple.root'),
                               # save only events passing the full path
                               SelectEvents   = cms.untracked.PSet( SelectEvents = cms.vstring('p') ),
                               # save PAT Layer 1 output; you need a '*' to
                               # unpack the list of commands 'patEventContent'
                               outputCommands = cms.untracked.vstring('drop *', *patEventContent )
                               )

process.outpath = cms.EndPath(process.out)
