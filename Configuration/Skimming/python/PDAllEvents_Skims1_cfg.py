# The following comments couldn't be translated into the new config version:

#looper = IterateNTimesLooper {uint32 nTimes = 50 }
#service = SimpleMemoryCheck{}

import FWCore.ParameterSet.Config as cms

process = cms.Process("SKIM")
process.load("FWCore.MessageLogger.MessageLogger_cfi")

# JetMETAnalysis
# skim: METLOW_SKIM
process.load("JetMETAnalysis.METSkims.metLow_SkimPaths_cff")

# skim: 1JET_SKIM
process.load("JetMETAnalysis.JetSkims.onejet_SkimPaths_cff")

# skim: PHOTON_JET_SKIM
process.load("JetMETAnalysis.JetSkims.photonjets_SkimPaths_cff")

#  include "JetMETAnalysis/Configuration/data/JetMETAnalysis_OutputModules.cff"
process.load("JetMETAnalysis.METSkims.metLowOutputModuleAODSIM_cfi")

process.load("JetMETAnalysis.JetSkims.onejetOutputModuleAODSIM_cfi")

process.load("JetMETAnalysis.JetSkims.photonjetsOutputModuleAODSIM_cfi")

process.configurationMetadata = cms.untracked.PSet(
    version = cms.untracked.string('$Revision: 1.4 $'),
    name = cms.untracked.string('$Source: /cvs_server/repositories/CMSSW/CMSSW/Configuration/Skimming/data/PDAllEvents_Skims1.cfg,v $'),
    annotation = cms.untracked.string('skims to be run on PDAllEvents 2/2')
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

process.JetMETAnalysis = cms.EndPath(process.metLowOutputModuleAODSIM+process.onejetOutputModuleAODSIM+process.photonjetsOutputModuleAODSIM)

