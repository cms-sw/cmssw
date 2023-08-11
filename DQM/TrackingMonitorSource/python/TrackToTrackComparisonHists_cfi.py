import FWCore.ParameterSet.Config as cms

from DQM.TrackingMonitorSource.trackToTrackComparisonHists_cfi import trackToTrackComparisonHists
from Configuration.Eras.Modifier_run3_common_cff import run3_common

run3_common.toModify(trackToTrackComparisonHists.histoPSet, onlinelumi_nbin=375, onlinelumi_rangeMin=200., onlinelumi_rangeMax=25000.)
run3_common.toModify(trackToTrackComparisonHists.histoPSet, PU_nbin=200, PU_rangeMin=0., PU_rangeMax=200.)