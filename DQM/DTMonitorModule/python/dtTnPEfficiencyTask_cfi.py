import FWCore.ParameterSet.Config as cms

from DQMServices.Core.DQMEDAnalyzer import DQMEDAnalyzer
dtTnPEfficiencyMonitor = DQMEDAnalyzer('DTTnPEfficiencyTask',
                                       # The muon object input tag
                                       inputTagMuons = cms.untracked.InputTag('muons'),
                                       # A string-based cut on muon variables
                                       probeCut = cms.untracked.string('isGlobalMuon'),
                                       # If true, enables detailed analysis plots
                                       detailedAnalysis = cms.untracked.bool(True)
)
