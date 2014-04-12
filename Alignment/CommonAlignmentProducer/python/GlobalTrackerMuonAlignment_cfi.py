import FWCore.ParameterSet.Config as cms

GlobalTrackerMuonAlignment = cms.EDAnalyzer('GlobalTrackerMuonAlignment',
                                            isolated = cms.bool(False),
                                            cosmics = cms.bool(False),
                                            refitmuon = cms.bool(False),
                                            refittrack = cms.bool(False),
                                            rootOutFile = cms.untracked.string('outfile.root'),
                                            txtOutFile = cms.untracked.string('outglobal.txt'),
                                            writeDB = cms.untracked.bool(False),
                                            debug = cms.untracked.bool(False)
)
