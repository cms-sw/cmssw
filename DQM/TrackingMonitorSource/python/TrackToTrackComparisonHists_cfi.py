import FWCore.ParameterSet.Config as cms

from DQM.TrackingMonitorSource.trackToTrackComparisonHists_cfi import trackToTrackComparisonHists
from Configuration.Eras.Modifier_run3_common_cff import run3_common
from Configuration.Eras.Modifier_phase2_common_cff import phase2_common

TrackToTrackComparisonHists = trackToTrackComparisonHists.clone()

run3_common.toModify(TrackToTrackComparisonHists.histoPSet, onlinelumi_nbin=375, onlinelumi_rangeMin=200., onlinelumi_rangeMax=25000.)
phase2_common.toModify(TrackToTrackComparisonHists.histoPSet, PU_nbin=200, PU_rangeMin=0., PU_rangeMax=200.)