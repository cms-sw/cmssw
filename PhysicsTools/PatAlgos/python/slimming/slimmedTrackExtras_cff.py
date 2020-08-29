from RecoMuon.MuonIdentification.muonReducedTrackExtras_cfi import muonReducedTrackExtras

import FWCore.ParameterSet.Config as cms

standAloneMuonReducedTrackExtras = muonReducedTrackExtras.clone(muonTag = "selectedPatMuons",
                                                                trackExtraTags = ["standAloneMuons"],
                                                                cut = "pt > 4.5",
                                                                outputClusters = False)

slimmedTrackExtrasTask = cms.Task(standAloneMuonReducedTrackExtras)
