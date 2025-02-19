# il file di iuput /tmp/noli/dimuons_allevt.root su lxplus204 

import FWCore.ParameterSet.Config as cms
import copy

process = cms.Process("Lavezzi")

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
    "file:dimuons_allevt.root"
    # "file:dimuons_1000evt.root"
    
    )
)

process.TFileService = cms.Service(
    "TFileService",
    fileName = cms.string("TriggerEfficiencyStudy.root")
)


zSelection = cms.PSet(
 cut = cms.string("charge = 0 & daughter(0).pt > 20 & daughter(1).pt > 20 & abs(daughter(0).eta)<2 & abs(daughter(1).eta)<2 & mass > 20 & mass < 200"),
    isoCut = cms.double(3.0),
    muonIsolations1 = cms.InputTag("muonIsolations"),  
    muonIsolations2 = cms.InputTag("muonIsolations")  
)


process.goodZToMuMu = cms.EDFilter(
    "ZToMuMuIsolatedSelector",
    zSelection,
    src = cms.InputTag("dimuonsGlobal"),
    filter = cms.bool(True) 
)


process.testAnalyzer = cms.EDAnalyzer(
    "testAnalyzer",
    selectMuon = cms.InputTag("selectedLayer1MuonsTriggerMatch"),
    ZMuMu = cms.InputTag("goodZToMuMu"),
    pathName = cms.string("HLT_Mu9"),
    EtaBins = cms.int32(40),
    minEta = cms.double(-2.),
    maxEta = cms.double(2.),
    PtBins = cms.int32(10),
    minPt = cms.double(20.),
    maxPt = cms.double(100.),
    EtaPt80Bins = cms.int32(10)
    )


process.eventInfo = cms.OutputModule (
    "AsciiOutputModule"
)




process.path = cms.Path (
    process.goodZToMuMu + 
    process.testAnalyzer
)


  
process.endPath = cms.EndPath( 
    process.eventInfo 
)

