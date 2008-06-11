import FWCore.ParameterSet.Config as cms
import copy

process = cms.Process("ZToMuMuSkim")

process.options = cms.untracked.PSet(
    wantSummary = cms.untracked.bool(True)
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(20)
)

process.source = cms.Source(
    "PoolSource",
    fileNames = cms.untracked.vstring(
    "file:/data1/cmsdata/dimuons/dimuons_job1.root",
    "file:/data1/cmsdata/dimuons/dimuons_job2.root",
    "file:/data1/cmsdata/dimuons/dimuons_job3.root"
    )
)

zSelection = cms.PSet(
    cut = cms.string(
      "daughter(0).pt>20 & daughter(1).pt>20 & abs(daughter(0).eta)<2 & abs(daughter(1).eta)<2 & mass>20 & (daughter(0).isGlobalMuon > 0.5|daughter(1).isGlobalMuon > 0.5)"
    )
)

process.selectedZToMuMu = cms.EDFilter(
    "CandViewRefSelector",
    zSelection,
    src = cms.InputTag("dimuons"),
    filter = cms.bool(True)
)

process.selectedZToMuMuOneTrack = cms.EDFilter(
    "CandViewRefSelector",
    zSelection,
    src = cms.InputTag("dimuonsOneTrack"),
    filter = cms.bool(True)
)

process.eventInfo = cms.OutputModule (
    "AsciiOutputModule"
)

process.out = cms.OutputModule (
    "PoolOutputModule",
    fileName = cms.untracked.string("zMuMuSkimOut.root"),
    outputCommands = cms.untracked.vstring(
      "keep *_genParticles_*_*",
      "keep L1MuRegionalCands_*_*_*",
      "keep L1MuGMTCands_*_*_*",
      "keep L1MuGMTReadoutCollection_*_*_*",
      "keep TriggerResults_*_*_*",
      "keep recoTracks_generalTracks_*_*",
      "keep recoTracks_globalMuons_*_*",
      "keep recoTracks_standAloneMuons_*_*",
      "keep recoMuons_muons_*_*",
      "keep *_selectedLayer1Muons_*_*",
      "keep *_goodTracks_*_*",
      "keep *_tkIsoDepositCalByAssociatorTowers_*_*",
      "keep *_EcalIsolationForTracks_*_*",
      "keep *_HcalIsolationForTracks_*_*",
      "keep *_goodMuonIsolations_*_*",
      "keep *_goodTrackIsolations_*_*",
      "keep *_muonIsolations_*_*",
      "keep *_dimuons_*_*",
      "keep *_dimuonsOneTrack_*_*",
      "keep *_dimuonsGlobal_*_*",
      "keep *_dimuonsOneStandAloneMuon_*_*",
      "keep *_muonMatch_*_*",
      "keep *_allDimuonsMCMatch_*_*",
      "keep *_muonHLTMatchHLT1MuonIso_*_*",
      "keep *_muonHLTMatchHLT1MuonNonIso_*_*",
      "keep *_muonHLTMatchHLT2MuonNonIso_*_*"       
    ),
    SelectEvents = cms.untracked.PSet(
      SelectEvents = cms.vstring(
        "zToMuMuFilterPath", "zToMuMuOneTrackfilterPath"
      )
   )
)

process.zToMuMuFilterPath = cms.Path(
    process.selectedZToMuMu
)

process.zToMuMuOneTrackfilterPath = cms.Path(
    process.selectedZToMuMuOneTrack 
)

process.endPath = cms.EndPath(
    process.eventInfo *
    process.out
)
