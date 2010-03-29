import FWCore.ParameterSet.Config as cms

GlobalTrackerMuonAlignment = cms.EDAnalyzer('GlobalTrackerMuonAlignment',
# can be redefined in GlobalTrackerMuonAlignment_test_cfg.py
                                            isolated = cms.bool(False),
                                            cosmics = cms.bool(False),
                                            rootOutFile = cms.untracked.string('outfile.root'),
                                            txtOutFile = cms.untracked.string('outglobal.txt'),
                                            writeDB = cms.untracked.bool(False),
                                            debug = cms.untracked.bool(False)
)
