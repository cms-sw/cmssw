import FWCore.ParameterSet.Config as cms

process = cms.Process("testJET")
process.load("FWCore.MessageService.MessageLogger_cfi")

process.load("SimGeneral.HepPDTESSource.pythiapdt_cfi")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring('/store/relval/2008/5/20/RelVal-RelValTTbar-1211209682-FakeConditions-2nd/0000/08765709-5826-DD11-9CE8-000423D94700.root')
)

process.selectTracks = cms.EDFilter("TrackSelector",
    src = cms.InputTag("generalTracks"),
    cut = cms.string('pt > 0.9 & numberOfValidHits > 7 & d0 <= 3.5 & dz <= 30')
)

process.allTracks = cms.EDProducer("ConcreteChargedCandidateProducer",
    src = cms.InputTag("selectTracks"),
    particleType = cms.string('pi+')
)

process.trackJet = cms.EDProducer("IterativeConeJetProducer",
    src = cms.InputTag("allTracks"),
    verbose = cms.untracked.bool(False),
    inputEtMin = cms.double(0.0),
    coneRadius = cms.double(0.5),
    alias = cms.untracked.string('IC5TracksJet'),
    seedThreshold = cms.double(1.0),
    debugLevel = cms.untracked.int32(0),
    jetType = cms.untracked.string('BasicJet'),
    inputEMin = cms.double(0.0)
)

process.printEvent = cms.EDAnalyzer("printTrackJet",
    src = cms.InputTag("trackJet")
)

process.printEventNumber = cms.OutputModule("AsciiOutputModule")

process.p = cms.Path(process.selectTracks*process.allTracks*process.trackJet*process.printEvent)
process.outpath = cms.EndPath(process.printEventNumber)
process.MessageLogger.cerr.default.limit = 1


