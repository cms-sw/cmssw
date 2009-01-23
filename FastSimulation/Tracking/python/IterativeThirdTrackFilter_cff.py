import FWCore.ParameterSet.Config as cms

##OLD WAY
##import RecoParticleFlow.PFTracking.vertexFilter_cfi
##thStep = RecoParticleFlow.PFTracking.vertexFilter_cfi.vertFilter.clone()
##iterativeThirdTrackFiltering = cms.Sequence(thStep)
##thStep.recTracks = cms.InputTag("iterativeThirdTrackMerging")
##thStep.TrackAlgorithm = 'iter3'
##thStep.DistZFromVertex = 0.1

import RecoTracker.FinalTrackSelectors.selectHighPurity_cfi
thStepVtx = RecoTracker.FinalTrackSelectors.selectHighPurity_cfi.selectHighPurity.clone()
thStepVtx.src = 'iterativeThirdTrackMerging'
thStepVtx.copyTrajectories = True
thStepVtx.chi2n_par = 0.9
thStepVtx.res_par = ( 0.003, 0.001 )
thStepVtx.d0_par1 = ( 0.9, 3.0 )
thStepVtx.dz_par1 = ( 0.9, 3.0 )
thStepVtx.d0_par2 = ( 1.0, 3.0 )
thStepVtx.dz_par2 = ( 1.0, 3.0 )

thStepTrk = RecoTracker.FinalTrackSelectors.selectHighPurity_cfi.selectHighPurity.clone()
thStepTrk.src = 'iterativeThirdTrackMerging'
thStepTrk.copyTrajectories = True
thStepTrk.chi2n_par = 0.5
thStepTrk.res_par = ( 0.003, 0.001 )
thStepTrk.minNumberLayers = 5
thStepTrk.d0_par1 = ( 1.0, 4.0 )
thStepTrk.dz_par1 = ( 1.0, 4.0 )
thStepTrk.d0_par2 = ( 1.0, 4.0 )
thStepTrk.dz_par2 = ( 1.0, 4.0 )

##import RecoTracker.FinalTrackSelectors.ctfrsTrackListMerger_cfi
##thStep = RecoTracker.FinalTrackSelectors.ctfrsTrackListMerger_cfi.ctfrsTrackListMerger.clone()
##thStep.TrackProducer1 = 'thStepVtx'
##thStep.TrackProducer2 = 'thStepTrk'


thStep = cms.EDFilter("FastTrackMerger",
                      TrackProducers = cms.VInputTag(cms.InputTag("thStepVtx"),
                                                     cms.InputTag("thStepTrk"))
)

iterativeThirdTrackFiltering = cms.Sequence(thStepVtx*thStepTrk*thStep)
