from RecoMuon.MuonIdentification.muonReducedTrackExtras_cfi import muonReducedTrackExtras

import FWCore.ParameterSet.Config as cms

slimmedMuonTrackExtras = muonReducedTrackExtras.clone(muonTag = "selectedPatMuons",
                                                                trackExtraTags = ["muonReducedTrackExtras", "standAloneMuons"],
                                                                trackExtraAssocs = ["muonReducedTrackExtras"],
                                                                pixelClusterTag = "muonReducedTrackExtras",
                                                                stripClusterTag = "muonReducedTrackExtras")

# no clusters in fastsim
from Configuration.Eras.Modifier_fastSim_cff import fastSim
fastSim.toModify(slimmedMuonTrackExtras, outputClusters = False)

# cluster collections are different in phase 2, so skip this for now
from Configuration.Eras.Modifier_phase2_tracker_cff import phase2_tracker
phase2_tracker.toModify(slimmedMuonTrackExtras, outputClusters = False)

# full set of track extras not available in existing AOD
from Configuration.Eras.Modifier_run2_miniAOD_80XLegacy_cff import run2_miniAOD_80XLegacy
run2_miniAOD_80XLegacy.toModify(slimmedMuonTrackExtras,
                                trackExtraTags = ["standAloneMuons"],
                                trackExtraAssocs = [],
                                outputClusters = False)

from Configuration.Eras.Modifier_run2_miniAOD_94XFall17_cff import run2_miniAOD_94XFall17
run2_miniAOD_94XFall17.toModify(slimmedMuonTrackExtras,
                                trackExtraTags = ["standAloneMuons"],
                                trackExtraAssocs = [],
                                outputClusters = False)

from Configuration.Eras.Modifier_pp_on_AA_2018_cff import pp_on_AA_2018
pp_on_AA_2018.toModify(slimmedMuonTrackExtras,
                                trackExtraTags = ["standAloneMuons"],
                                trackExtraAssocs = [],
                                outputClusters = False)

from Configuration.ProcessModifiers.run2_miniAOD_UL_cff import run2_miniAOD_UL
run2_miniAOD_UL.toModify(slimmedMuonTrackExtras,
                                trackExtraTags = ["standAloneMuons"],
                                trackExtraAssocs = [],
                                outputClusters = False)
