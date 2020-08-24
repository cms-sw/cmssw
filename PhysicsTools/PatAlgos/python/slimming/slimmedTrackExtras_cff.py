from RecoMuon.MuonIdentification.muonReducedTrackExtras_cfi import muonReducedTrackExtras

import FWCore.ParameterSet.Config as cms

standAloneMuonReducedTrackExtras = muonReducedTrackExtras.clone(trackExtraTags = ["standAloneMuons"],
                                                                outputClusters = False)

slimmedTrackExtrasTask = cms.Task(standAloneMuonReducedTrackExtras)
