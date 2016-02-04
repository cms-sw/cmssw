###########################
#                         #
# author: Pasquale Noli   #
# INFN Naples             #
# Script to run gamma     #
# analysis                #   
#                         #
###########################



import FWCore.ParameterSet.Config as cms
import copy

process = cms.Process("Pocho")

process.include("FWCore/MessageLogger/data/MessageLogger.cfi")

process.options = cms.untracked.PSet(
    wantSummary = cms.untracked.bool(False)
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(10000)
    )

process.source = cms.Source(
    "PoolSource",
    fileNames = cms.untracked.vstring(
    
    "file:/data1/home/noli/roofile_SUMMER08/zMuMu_dimuons_1.root",
    "file:/data1/home/noli/roofile_SUMMER08/zMuMu_dimuons_2.root",
    "file:/data1/home/noli/roofile_SUMMER08/zMuMu_dimuons_3.root",
    "file:/data1/home/noli/roofile_SUMMER08/zMuMu_dimuons_4.root",
    "file:/data1/home/noli/roofile_SUMMER08/zMuMu_dimuons_6.root",
    "file:/data1/home/noli/roofile_SUMMER08/zMuMu_dimuons_7.root",
    "file:/data1/home/noli/roofile_SUMMER08/zMuMu_dimuons_8.root",
    "file:/data1/home/noli/roofile_SUMMER08/zMuMu_dimuons_9.root",
    "file:/data1/home/noli/roofile_SUMMER08/zMuMu_dimuons_10.root"

    )
)

process.TFileService = cms.Service(
    "TFileService",
    fileName = cms.string("gamma_analysis.root")
)




#ZMuSta
process.goodZToMuMuOneStandAloneMuon = cms.EDFilter(
    "ZMuMuOverlapExclusionSelector",
    src = cms.InputTag("dimuonsOneStandAloneMuon"),
    overlap = cms.InputTag("dimuonsGlobal"),
    filter = cms.bool(False)
)


#ZMuTk
process.zToMuGlobalMuOneTrack = cms.EDFilter(
    "CandViewRefSelector",
    cut = cms.string("daughter(0).isGlobalMuon = 1"),
    src = cms.InputTag("dimuonsOneTrack"),
    filter = cms.bool(False)
)

process.goodZToMuMuOneTrack = cms.EDFilter(
   "ZMuMuOverlapExclusionSelector",
    src = cms.InputTag("zToMuGlobalMuOneTrack"),
    overlap = cms.InputTag("dimuonsGlobal"),
    filter = cms.bool(False)
)



process.Analyzer = cms.EDAnalyzer(
    "gamma_radiative_analyzer",
    zMuMu = cms.InputTag("dimuonsGlobal"),
    zMuMuMatchMap= cms.InputTag("allDimuonsMCMatch"),
    zMuTk = cms.InputTag("goodZToMuMuOneTrack"),
    zMuTkMatchMap= cms.InputTag("allDimuonsMCMatch"),
    zMuSa = cms.InputTag("goodZToMuMuOneStandAloneMuon"),
    zMuSaMatchMap= cms.InputTag("allDimuonsMCMatch"),
      )


process.eventInfo = cms.OutputModule (
    "AsciiOutputModule"
)

process.path = cms.Path (
    process.goodZToMuMuOneStandAloneMuon+
    process.zToMuGlobalMuOneTrack +
    process.goodZToMuMuOneTrack +
    process.Analyzer
)


  
