import FWCore.ParameterSet.Config as cms

process = cms.Process("SKIM")
process.load("FWCore.MessageLogger.MessageLogger_cfi")

# QCDAnalysis
# naming is mixed up 
process.load("QCDAnalysis.Skimming.qcdJetFilterStreamHiPath_cff")

process.load("QCDAnalysis.Skimming.qcdJetFilterStreamMedPath_cff")

process.load("QCDAnalysis.Skimming.qcdJetFilterStreamLoPath_cff")

# this is a bugfix, because QCD doesn't follow the guidelines
process.load("QCDAnalysis.Configuration.QCDAnalysis_EventContent_cff")

process.load("QCDAnalysis.Skimming.qcdJetFilterStreamHiOutputModule_cfi")

process.load("QCDAnalysis.Skimming.qcdJetFilterStreamMedOutputModule_cfi")

process.load("QCDAnalysis.Skimming.qcdJetFilterStreamLoOutputModule_cfi")

process.configurationMetadata = cms.untracked.PSet(
    version = cms.untracked.string('$Revision: 1.3 $'),
    name = cms.untracked.string('$Source: /cvs_server/repositories/CMSSW/CMSSW/Configuration/Skimming/data/PDJetMET_Skims2.cfg,v $'),
    annotation = cms.untracked.string('skims to be run on PDJetMET 1/2')
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
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring('/store/mc/2007/9/17/RelVal-W1jet_100ptw300-alpgen-1190037682/0002/0AFADF7A-3565-DC11-BCE6-000423D944DC.root')
)

process.QCDAnalysis = cms.EndPath(process.qcdJetFilterStreamHiOutputModule+process.qcdJetFilterStreamMedOutputModule+process.qcdJetFilterStreamLoOutputModule)

