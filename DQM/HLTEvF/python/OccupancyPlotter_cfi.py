import FWCore.ParameterSet.Config as cms

onlineOccPlot = cms.EDAnalyzer('OccupancyPlotter',

                      dirname = cms.untracked.string("HLT/OccupancyPlots/"),

                      
)
