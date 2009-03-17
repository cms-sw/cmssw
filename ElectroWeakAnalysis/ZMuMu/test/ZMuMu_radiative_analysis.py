# il file di iuput /tmp/noli/dimuons_allevt.root su lxplus204 

import FWCore.ParameterSet.Config as cms
import copy

process = cms.Process("Diegol")

process.include("FWCore/MessageLogger/data/MessageLogger.cfi")

process.options = cms.untracked.PSet(
    wantSummary = cms.untracked.bool(True)
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(100)
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
    fileName = cms.string("analisi_radiativi_overlap.root")
)


zSelection = cms.PSet(
cut = cms.string("charge = 0 & daughter(0).pt > 20 & daughter(1).pt > 20 & abs(daughter(0).eta)<2 & abs(daughter(1).eta)<2 & mass > 20 & mass < 200"),
  isoCut = cms.double(3.0),
  isolationType = cms.string("track")
)


process.goodZToMuMu = cms.EDFilter(
    "ZToMuMuIsolatedSelector",
    zSelection,
    src = cms.InputTag("dimuonsGlobal"),
    filter = cms.bool(True) 
)

#ZMuMu: richiedo almeno 1 HLT trigger match
process.goodZToMuMuHLT = cms.EDFilter(
    "ZHLTMatchFilter",
    src = cms.InputTag("goodZToMuMu"),
    condition =cms.string("atLeastOneMatched"),
    hltPath = cms.string("hltSingleMuNoIsoL3PreFiltered"),
    filter = cms.bool(True) 
)


process.zToMuGlobalMuOneTrack = cms.EDFilter(
    "CandViewRefSelector",
    cut = cms.string("daughter(0).isGlobalMuon = 1"),
    src = cms.InputTag("dimuonsOneTrack"),
    filter = cms.bool(True)
)

#process.goodZToMuMuOneTrack = cms.EDFilter(
#   "ZMuMuOverlapExclusionSelector",
#    src = cms.InputTag("zToMuGlobalMuOneTrack"),
#    overlap = cms.InputTag("goodZToMuMu"),
#    filter = cms.bool(True)
#)


#ZMuTk:richiedo che il muGlobal 'First' ha HLT match
process.goodZToMuMuOneTrackFirstHLT = cms.EDFilter(
    "ZHLTMatchFilter",
    # src = cms.InputTag("goodZToMuMuOneTrack"),
    src = cms.InputTag("zToMuGlobalMuOneTrack"),
    condition =cms.string("firstMatched"),
    hltPath = cms.string("hltSingleMuNoIsoL3PreFiltered"),
    filter = cms.bool(True) 
)


process.Analyzer = cms.EDAnalyzer(
    "ZMuMu_Radiative_analyzer",
    zMuMu = cms.InputTag("goodZToMuMuHLT"),
    zMuMuMatchMap= cms.InputTag("allDimuonsMCMatch"),
    zMuTk = cms.InputTag("goodZToMuMuOneTrackFirstHLT"),
    zMuTkMatchMap= cms.InputTag("allDimuonsMCMatch"),
    veto = cms.untracked.double(0.015),
    deltaRTrk = cms.untracked.double("0.3"),
    ptThreshold = cms.untracked.double("1.5")
    )


process.eventInfo = cms.OutputModule (
    "AsciiOutputModule"
)

process.path = cms.Path (
    process.goodZToMuMu +
    process.goodZToMuMuHLT +
    process.zToMuGlobalMuOneTrack +
   # process.goodZToMuMuOneTrack +
    process.goodZToMuMuOneTrackFirstHLT +
    process.Analyzer
)


  
process.endPath = cms.EndPath( 
    process.eventInfo 
)

