# The following comments couldn't be translated into the new config version:

#looper = IterateNTimesLooper {uint32 nTimes = 50 }
#service = SimpleMemoryCheck{}

import FWCore.ParameterSet.Config as cms

process = cms.Process("SKIM")
process.load("FWCore.MessageLogger.MessageLogger_cfi")

# MuonAnalysis
# skim: muon L1
process.load("MuonAnalysis.Configuration.muonL1_SkimPath_cff")

#  include "MuonAnalysis/Configuration/data/MuonAnalysis_OutputModules.cff"
process.load("MuonAnalysis.Configuration.muonL1OutputModuleRECOSIM_cfi")

# ParticleFlow
# skim: PF JPsiee
process.load("RecoParticleFlow.PFSkims.jpsiee_SkimPaths_cff")

#  include "RecoParticleFlow/PFSkims/data/PFSkims_OutputModules.cff"
process.load("RecoParticleFlow.PFSkims.jpsieeOutputModuleFEVTSIM_cfi")

# QCDAnalysis
# QCD Jet+X
process.load("QCDAnalysis.Configuration.QCDAnalysis_EventContent_cff")

process.load("QCDAnalysis.Skimming.softJetsPath_cff")

#  include "QCDAnalysis/Configuration/data/QCDAnalysis_OutputModules.cff"
process.load("QCDAnalysis.Skimming.softJetsOutputModule_cfi")

process.configurationMetadata = cms.untracked.PSet(
    version = cms.untracked.string('$Revision: 1.4 $'),
    name = cms.untracked.string('$Source: /cvs_server/repositories/CMSSW/CMSSW/Configuration/Skimming/data/PDAllEvents_Skims2.cfg,v $'),
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

process.MuonAnalysis = cms.EndPath(process.muonL1OutputModuleRECOSIM)
process.PFAnalysis = cms.EndPath(process.jpsieeOutputModuleFEVTSIM)
process.QCDAnalysis = cms.EndPath(process.softJetsOutputModule)

