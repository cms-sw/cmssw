from RecoMuon.MuonIdentification.muonReducedTrackExtras_cfi import muonReducedTrackExtras

import FWCore.ParameterSet.Config as cms

slimmedMuonTrackExtras = muonReducedTrackExtras.clone(muonTag = "selectedPatMuons",
                                                                cut = "pt > 4.5",
                                                                trackExtraTags = ["muonReducedTrackExtras", "standAloneMuons"],
                                                                trackExtraAssocs = ["muonReducedTrackExtras"],
                                                                pixelClusterTag = "muonReducedTrackExtras",
                                                                stripClusterTag = "muonReducedTrackExtras")

from RecoMuon.MuonIdentification.displacedMuonReducedTrackExtras_cfi import displacedMuonReducedTrackExtras
slimmedDisplacedMuonTrackExtras = displacedMuonReducedTrackExtras.clone(muonTag = "selectedPatDisplacedMuons",
                                                                cut = "pt > 4.5",
                                                                trackExtraTags = ["displacedMuonReducedTrackExtras", "displacedStandAloneMuons"],
                                                                trackExtraAssocs = ["displacedMuonReducedTrackExtras"],
                                                                pixelClusterTag = "displacedMuonReducedTrackExtras",
                                                                stripClusterTag = "displacedMuonReducedTrackExtras")

# no clusters in fastsim
from Configuration.Eras.Modifier_fastSim_cff import fastSim
fastSim.toModify(slimmedMuonTrackExtras, outputClusters = False)

# cluster collections are different in phase 2, so skip this for now
from Configuration.Eras.Modifier_phase2_tracker_cff import phase2_tracker
phase2_tracker.toModify(slimmedMuonTrackExtras, outputClusters = False)
phase2_tracker.toModify(slimmedDisplacedMuonTrackExtras, outputClusters = False)

# lower minimum pt for B-parking
from Configuration.Eras.Modifier_bParking_cff import bParking
bParking.toModify(slimmedMuonTrackExtras, cut = "pt > 3.0")

# full set of track extras not available in existing AOD
from Configuration.Eras.Modifier_run2_miniAOD_80XLegacy_cff import run2_miniAOD_80XLegacy
from Configuration.Eras.Modifier_run2_miniAOD_94XFall17_cff import run2_miniAOD_94XFall17
from Configuration.ProcessModifiers.pp_on_AA_cff import pp_on_AA
from Configuration.ProcessModifiers.miniAOD_skip_trackExtras_cff import miniAOD_skip_trackExtras

(run2_miniAOD_80XLegacy | run2_miniAOD_94XFall17 | pp_on_AA | miniAOD_skip_trackExtras).toModify(slimmedMuonTrackExtras,
                                trackExtraTags = ["standAloneMuons"],
                                trackExtraAssocs = [],
                                outputClusters = False)
