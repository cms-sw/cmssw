# The following comments couldn't be translated into the new config version:

# include "FWCore/MessageLogger/data/MessageLogger.cfi"

import FWCore.ParameterSet.Config as cms

process = cms.Process("SKIM")
# EgammaAnalysis
# skim: egamma WJets 
process.load("EgammaAnalysis.CSA07Skims.EgammaWPlusEMOrJetPath_cff")

process.load("EgammaAnalysis.CSA07Skims.egammaWPlusEMOrJetOutputModuleRECOSIM_cfi")

# TopQuarkAnalysis
# skim: topDiLeptonMuonX
process.load("TopQuarkAnalysis.TopSkimming.topDiLeptonMuonX_SkimPaths_cff")

# skim: topSemiLepMuon
process.load("TopQuarkAnalysis.TopSkimming.topSemiLepMuon_SkimPaths_cff")

process.load("TopQuarkAnalysis.TopSkimming.topDiLeptonMuonXOutputModule_cfi")

process.load("TopQuarkAnalysis.TopSkimming.topSemiLepMuonOutputModule_cfi")

process.configurationMetadata = cms.untracked.PSet(
    version = cms.untracked.string('$Revision: 1.3 $'),
    name = cms.untracked.string('$Source: /cvs_server/repositories/CMSSW/CMSSW/Configuration/Skimming/data/PDMuon_Skims4.cfg,v $'),
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

process.EgammaAnalysis = cms.EndPath(process.egammaWPlusEMOrJetOutputModuleRECOSIM)
process.TopQuarkAnalysis = cms.EndPath(process.topDiLeptonMuonXOutputModule+process.topSemiLepMuonOutputModule)

