import FWCore.ParameterSet.Config as cms

from DQMServices.Core.DQMEDAnalyzer import DQMEDAnalyzer
gemTnPEfficiencyMonitor = DQMEDAnalyzer('GEMTnPEfficiencyTask',
                                       # The muon object input tag
                                       inputTagMuons = cms.untracked.InputTag('muons'),
                                       inputTagPrimaryVertices = cms.untracked.InputTag('offlinePrimaryVertices'),
                                       trigResultsTag = cms.untracked.InputTag("TriggerResults::HLT"),
                                       trigEventTag = cms.untracked.InputTag("hltTriggerSummaryAOD::HLT"),
                                       # A string-based cut on muon variables
                                       probeCut = cms.untracked.string('isTrackerMuon && (innerTrack.normalizedChi2 < 10) && (innerTrack.hitPattern.numberOfValidPixelHits > 0) && (innerTrack.hitPattern.trackerLayersWithMeasurement > 5) && ((isolationR03.sumPt)/(pt) < 0.1) && pt>10.' ),
                                       probeDxyCut = cms.untracked.double(0.2),
                                       probeDzCut = cms.untracked.double(0.5),
                                       #Cut on muon ID:
				       #  CutBasedIdLoose = 1UL << 0
                                       #  CutBasedIdMedium = 1UL << 1
                                       #  CutBasedIdMediumPrompt = 1UL << 2 
				       #  CutBasedIdTight = 1UL << 3
                                       tagCut = cms.untracked.string('(selectors & 8) && ((isolationR03.sumPt)/(pt) < 0.05) && pt>24.'),
                                       borderCut = cms.untracked.double(-10.),
                                       lowPairMassCut = cms.untracked.double (80.),
                                       highPairMassCut = cms.untracked.double (100.),
                                       trigName = cms.untracked.string("HLT_IsoMu*"),
                                       #cuts for passing probe definition
                                       dx_cut = cms.untracked.double(10.),
                                       # If true, enables detailed analysis plots
                                       detailedAnalysis = cms.untracked.bool(True)
)
