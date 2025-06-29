import FWCore.ParameterSet.Config as cms

from DQM.TrackingMonitorSource.trackToTrackComparisonHists_cfi import trackToTrackComparisonHists
from Configuration.Eras.Modifier_run3_common_cff import run3_common
from Configuration.Eras.Modifier_phase2_common_cff import phase2_common

TrackToTrackComparisonHists = trackToTrackComparisonHists.clone()

from Configuration.Eras.Modifier_stage2L1Trigger_cff import stage2L1Trigger
stage2L1Trigger.toModify(TrackToTrackComparisonHists,
                         genericTriggerEventPSet = dict(stage2 = cms.bool(True),
                                                        l1tAlgBlkInputTag = cms.InputTag("gtStage2Digis"),
                                                        l1tExtBlkInputTag = cms.InputTag("gtStage2Digis"),
                                                        ReadPrescalesFromFile = cms.bool(False))
                         )

run3_common.toModify(TrackToTrackComparisonHists.histoPSet, Eta_rangeMin=-3.,Eta_rangeMax =3.)
run3_common.toModify(TrackToTrackComparisonHists.histoPSet, onlinelumi_nbin=375, onlinelumi_rangeMin=200., onlinelumi_rangeMax=25000.)
phase2_common.toModify(TrackToTrackComparisonHists.histoPSet, Eta_rangeMin=-4.,Eta_rangeMax =4.)
phase2_common.toModify(TrackToTrackComparisonHists.histoPSet, PU_nbin=200, PU_rangeMin=0., PU_rangeMax=200.)
