# The following comments couldn't be translated into the new config version:

#looper = IterateNTimesLooper {uint32 nTimes = 50 }
#service = SimpleMemoryCheck{}

import FWCore.ParameterSet.Config as cms

process = cms.Process("SKIM")
process.load("FWCore.MessageLogger.MessageLogger_cfi")

# RecoBTag
# skim: btagDijet
process.load("RecoBTag.Skimming.btagDijet_SkimPaths_cff")

# skim: btagElecInJet
process.load("RecoBTag.Skimming.btagElecInJet_SkimPaths_cff")

# skim: btagMuonInJet
process.load("RecoBTag.Skimming.btagMuonInJet_SkimPaths_cff")

#  include "RecoBTag/Skimming/data/RecoBTag_OutputModules.cff"
process.load("RecoBTag.Skimming.btagDijetOutputModuleAODSIM_cfi")

process.load("RecoBTag.Skimming.btagElecInJetOutputModuleAODSIM_cfi")

process.load("RecoBTag.Skimming.btagMuonInJetOutputModuleAODSIM_cfi")

process.configurationMetadata = cms.untracked.PSet(
    version = cms.untracked.string('$Revision: 1.5 $'),
    name = cms.untracked.string('$Source: /cvs_server/repositories/CMSSW/CMSSW/Configuration/Skimming/data/PDAllEvents_Skims0.cfg,v $'),
    annotation = cms.untracked.string('skims to be run on PDAllEvents')
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
    fileNames = cms.untracked.vstring('/store/CSA07/mc/2007/10/2/CSA07-CSA07AllEvents-Tier0-A2-Chowder/0000/002F6479-F371-DC11-BEB5-000423D94A04.root')
)

process.RecoBTag = cms.EndPath(process.btagDijetOutputModuleAODSIM+process.btagElecInJetOutputModuleAODSIM+process.btagMuonInJetOutputModuleAODSIM)

