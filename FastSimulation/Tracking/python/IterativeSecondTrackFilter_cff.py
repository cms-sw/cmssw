import FWCore.ParameterSet.Config as cms

##import RecoParticleFlow.PFTracking.vertexFilter_cfi
##secStep = RecoParticleFlow.PFTracking.vertexFilter_cfi.vertFilter.clone()
##iterativeSecondTrackFiltering = cms.Sequence(secStep)
##secStep.recTracks = cms.InputTag("iterativeSecondTrackMerging")
##secStep.TrackAlgorithm = 'iter2'

# track selection
import RecoTracker.FinalTrackSelectors.selectHighPurity_cfi

secStepVtx = RecoTracker.FinalTrackSelectors.selectHighPurity_cfi.selectHighPurity.clone(
src = 'iterativeSecondTrackMerging',
copyTrajectories = True,
copyExtras = True,
chi2n_par = 0.9,
res_par = ( 0.003, 0.001 ),
minNumberLayers = 3,
minNumber3DLayers = 3,
maxNumberLostLayers = 1,
d0_par1 = ( 0.85, 3.0 ),
dz_par1 = ( 0.8, 3.0 ),
d0_par2 = ( 0.9, 3.0 ),
dz_par2 = ( 0.9, 3.0 )
)

secStepTrk = RecoTracker.FinalTrackSelectors.selectHighPurity_cfi.selectHighPurity.clone(
src = 'iterativeSecondTrackMerging',
copyTrajectories = True,
copyExtras = True,
chi2n_par = 0.5,
res_par = ( 0.003, 0.001 ),
minNumberLayers = 5,
minNumber3DLayers = 3,
maxNumberLostLayers = 1,
d0_par1 = ( 0.9, 4.0 ),
dz_par1 = ( 0.9, 4.0 ),
d0_par2 = ( 0.9, 4.0 ),
dz_par2 = ( 0.9, 4.0 )
)

##import RecoTracker.FinalTrackSelectors.ctfrsTrackListMerger_cfi
##secStep = RecoTracker.FinalTrackSelectors.ctfrsTrackListMerger_cfi.ctfrsTrackListMerger.clone()
##secStep.TrackProducer1 = 'secStepVtx'
##secStep.TrackProducer2 = 'secStepTrk'

secStep = cms.EDProducer("FastTrackMerger",
                       TrackProducers = cms.VInputTag(cms.InputTag("secStepVtx"),
                                                     cms.InputTag("secStepTrk"))
)

secfilter = cms.EDProducer("QualityFilter",
    TrackQuality = cms.string('highPurity'),
    recTracks = cms.InputTag("secStep")
)

iterativeSecondTrackFiltering = cms.Sequence(secStepVtx*secStepTrk*secStep*secfilter)
