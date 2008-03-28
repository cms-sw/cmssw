import FWCore.ParameterSet.Config as cms

from AnalysisExamples.SiStripDetectorPerformance.TrackingEff_EventContent_cff import *
from Configuration.EventContent.EventContent_cff import *
AODSIMTrackingEffEventContent = cms.PSet(
    outputCommands = cms.untracked.vstring()
)
TrackingEffOutputModule = cms.OutputModule("PoolOutputModule",
    TrackingEffEventSelection,
    AODSIMTrackingEffEventContent,
    dataset = cms.untracked.PSet(
        filterName = cms.untracked.string('TrackingEffAODSIM'),
        dataTier = cms.untracked.string('USER')
    ),
    fileName = cms.untracked.string('TrackingEff.root')
)

AODSIMTrackingEffEventContent.outputCommands.extend(AODSIMEventContent.outputCommands)
AODSIMTrackingEffEventContent.outputCommands.extend(TrackingEffEventContent.outputCommands)

