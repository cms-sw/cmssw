import FWCore.ParameterSet.Config as cms

'''

reskimForTraining

A second layer of skim to further select only those events that have a tau
candidate with pT above a given threshold.  This stage also facilitates merging
many files into fewer ones.

Author: Evan K. Friis (UC Davis)

'''

process = cms.Process("TANCreskim")

maxEvents = cms.untracked.PSet( input = cms.untracked.int32(-1) )
readFiles = cms.untracked.vstring()
secFiles = cms.untracked.vstring()
process.source = cms.Source("PoolSource", fileNames = readFiles,
                            secondaryFileNames = secFiles)

_RECO_TAU_CUT = "pt > 10 & abs(eta) < 2.5"

process.signalExists = cms.EDFilter(
    "CandCollectionExistFilter",
    src = cms.InputTag("hpsTancTausPassingDecayModeSignal"),
)
process.signalPassingPtThreshold = cms.EDFilter(
    "PFTauViewRefSelector",
    src = cms.InputTag("hpsTancTausPassingDecayModeSignal"),
    cut = cms.string(_RECO_TAU_CUT),
    filter = cms.bool(True),
)

process.signalPath = cms.Path(
    process.signalExists*
    process.signalPassingPtThreshold
)

process.backgroundExists = cms.EDFilter(
    "CandCollectionExistFilter",
    src = cms.InputTag("hpsTancTausPassingDecayModeBackground"),
)
process.backgroundPassingPtThreshold = cms.EDFilter(
    "PFTauViewRefSelector",
    src = cms.InputTag("hpsTancTausPassingDecayModeBackground"),
    cut = cms.string(_RECO_TAU_CUT),
    filter = cms.bool(True),
)

process.backgroundPath = cms.Path(
    process.backgroundExists*
    process.backgroundPassingPtThreshold
)

process.write = cms.OutputModule(
    "PoolOutputModule",
    fileName = cms.untracked.string("selected_events.root"),
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring("signalPath", "backgroundPath"),
    ),
)
process.out = cms.EndPath(process.write)

process.schedule = cms.Schedule(
    process.signalPath,
    process.backgroundPath,
    process.out
)

process.options = cms.untracked.PSet( wantSummary = cms.untracked.bool(True) )
