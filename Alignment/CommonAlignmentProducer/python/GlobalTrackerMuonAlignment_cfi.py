import FWCore.ParameterSet.Config as cms

GlobalTrackerMuonAlignment = cms.EDAnalyzer('GlobalTrackerMuonAlignment',
                                            cosmics = cms.bool(False),
                                            isolated = cms.bool(False),
                                            rootOutFile = cms.untracked.string('outfile.root'),
                                            txtOutFile = cms.untracked.string('outglobal.txt'),
                                            writeDB = cms.untracked.bool(False),
                                            debug = cms.untracked.bool(False)
)
