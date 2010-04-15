# The following comments couldn't be translated into the new config version:

# include "FWCore/MessageLogger/data/MessageLogger.cfi"

import FWCore.ParameterSet.Config as cms

process = cms.Process("SKIM")
# SUSYBSMAnalysis
# for hscp refitting
process.load("Configuration.StandardSequences.FakeConditions_cff")

process.load("Geometry.TrackerGeometryBuilder.trackerGeometry_cfi")

process.load("Configuration.StandardSequences.Reconstruction_cff")

# skim: SUSYBSM muon
process.load("SUSYBSMAnalysis.Skimming.SusyMuon_SkimPaths_cff")

# skim: SUSYBSM muons hits
process.load("SUSYBSMAnalysis.Skimming.SusyMuonHits_SkimPaths_cff")

# skim: SUSYBSM muon HLT
process.load("SUSYBSMAnalysis.Skimming.SusyMuon_HLT_SkimPaths_cff")

process.load("SUSYBSMAnalysis.Skimming.susyMuonOutputModule_cfi")

process.load("SUSYBSMAnalysis.Skimming.susyMuonHitsOutputModule_cfi")

process.load("SUSYBSMAnalysis.Skimming.susyHLTMuonOutputModule_cfi")

process.configurationMetadata = cms.untracked.PSet(
    version = cms.untracked.string('$Revision: 1.3 $'),
    name = cms.untracked.string('$Source: /cvs_server/repositories/CMSSW/CMSSW/Configuration/Skimming/data/PDMuon_Skims6.cfg,v $'),
    annotation = cms.untracked.string('skims to be run on PDMuon')
)
process.options = cms.untracked.PSet(
    FailPath = cms.untracked.vstring('ProductNotFound'),
    makeTriggerResults = cms.untracked.bool(True),
    wantSummary = cms.untracked.bool(True)
)
process.LoadAllDictionaries = cms.Service("LoadAllDictionaries")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)
process.MessageLogger = cms.Service("MessageLogger",
    cout = cms.untracked.PSet(
        threshold = cms.untracked.string('INFO')
    ),
    destinations = cms.untracked.vstring('cout')
)

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring('/store/mc/2007/9/17/RelVal-W1jet_100ptw300-alpgen-1190037682/0002/0AFADF7A-3565-DC11-BCE6-000423D944DC.root')
)

process.SUSYBSMAnalysis = cms.EndPath(process.susyMuonOutputModule+process.susyMuonHitsOutputModule+process.susyHLTMuonOutputModule)

