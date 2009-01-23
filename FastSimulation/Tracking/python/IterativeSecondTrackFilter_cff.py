import FWCore.ParameterSet.Config as cms

##import RecoParticleFlow.PFTracking.vertexFilter_cfi
##secStep = RecoParticleFlow.PFTracking.vertexFilter_cfi.vertFilter.clone()
##iterativeSecondTrackFiltering = cms.Sequence(secStep)
##secStep.recTracks = cms.InputTag("iterativeSecondTrackMerging")
##secStep.TrackAlgorithm = 'iter2'

# track selection
import RecoTracker.FinalTrackSelectors.selectHighPurity_cfi

secStepVtx = RecoTracker.FinalTrackSelectors.selectHighPurity_cfi.selectHighPurity.clone()
secStepVtx.src = 'iterativeSecondTrackMerging'
secStepVtx.copyTrajectories = True
secStepVtx.chi2n_par = 0.9
secStepVtx.res_par = ( 0.003, 0.001 )
secStepVtx.d0_par1 = ( 0.85, 3.0 )
secStepVtx.dz_par1 = ( 0.8, 3.0 )
secStepVtx.d0_par2 = ( 0.9, 3.0 )
secStepVtx.dz_par2 = ( 0.9, 3.0 )

secStepTrk = RecoTracker.FinalTrackSelectors.selectHighPurity_cfi.selectHighPurity.clone()
secStepTrk.src = 'iterativeSecondTrackMerging'
secStepTrk.copyTrajectories = True
secStepTrk.chi2n_par = 0.5
secStepTrk.res_par = ( 0.003, 0.001 )
secStepTrk.minNumberLayers = 5
secStepTrk.d0_par1 = ( 0.9, 4.0 )
secStepTrk.dz_par1 = ( 0.9, 4.0 )
secStepTrk.d0_par2 = ( 0.9, 4.0 )
secStepTrk.dz_par2 = ( 0.9, 4.0 )

##import RecoTracker.FinalTrackSelectors.ctfrsTrackListMerger_cfi
##secStep = RecoTracker.FinalTrackSelectors.ctfrsTrackListMerger_cfi.ctfrsTrackListMerger.clone()
##secStep.TrackProducer1 = 'secStepVtx'
##secStep.TrackProducer2 = 'secStepTrk'

secStep = cms.EDFilter("FastTrackMerger",
                       TrackProducers = cms.VInputTag(cms.InputTag("secStepVtx"),
                                                     cms.InputTag("secStepTrk"))
)

iterativeSecondTrackFiltering = cms.Sequence(secStepVtx*secStepTrk*secStep)
