###########################
#                         #
# author: Pasquale Noli   #
# INFN Naples             #
# Script to run radiative #
# analysis                #   
#                         #
###########################



import FWCore.ParameterSet.Config as cms
import copy

process = cms.Process("Diegol")

process.include("FWCore/MessageLogger/data/MessageLogger.cfi")

process.options = cms.untracked.PSet(
    wantSummary = cms.untracked.bool(False)
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
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
    fileName = cms.string("analysis_radiative_table.root")
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


process.goodZToMuMuOneTrackFirstHLT = cms.EDFilter(
    "ZHLTMatchFilter",
    src = cms.InputTag("goodZToMuMuOneTrack"),
    condition =cms.string("firstMatched"),
    hltPath = cms.string("hltSingleMuNoIsoL3PreFiltered"),
    filter = cms.bool(False) 
)


process.Analyzer = cms.EDAnalyzer(
    "ZMuMu_Radiative_analyzer",
    zMuMu = cms.InputTag("dimuonsGlobal"),
    zMuMuMatchMap= cms.InputTag("allDimuonsMCMatch"),
    zMuTk = cms.InputTag("goodZToMuMuOneTrackFirstHLT"),
    zMuTkMatchMap= cms.InputTag("allDimuonsMCMatch"),
    zMuSa = cms.InputTag("goodZToMuMuOneStandAloneMuon"),
    zMuSaMatchMap= cms.InputTag("allDimuonsMCMatch"),
    veto = cms.untracked.double(0.015),
    deltaRTrk = cms.untracked.double(0.3),
    ptThreshold = cms.untracked.double(1.5)
    )


process.eventInfo = cms.OutputModule (
    "AsciiOutputModule"
)

process.path = cms.Path (
    process.goodZToMuMuOneStandAloneMuon+
    process.zToMuGlobalMuOneTrack +
    process.goodZToMuMuOneTrack +
    process.goodZToMuMuOneTrackFirstHLT +
    process.Analyzer
)


  
#process.endPath = cms.EndPath( 
#    process.eventInfo 
#)

