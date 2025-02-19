import FWCore.ParameterSet.Config as cms

##OLD WAY
##import RecoParticleFlow.PFTracking.vertexFilter_cfi
##thStep = RecoParticleFlow.PFTracking.vertexFilter_cfi.vertFilter.clone()
##iterativeThirdTrackFiltering = cms.Sequence(thStep)
##thStep.recTracks = cms.InputTag("iterativeThirdTrackMerging")
##thStep.TrackAlgorithm = 'iter3'
##thStep.DistZFromVertex = 0.1

import RecoTracker.FinalTrackSelectors.selectHighPurity_cfi
thStepVtx = RecoTracker.FinalTrackSelectors.selectHighPurity_cfi.selectHighPurity.clone(
src = 'iterativeThirdTrackMerging',
copyTrajectories = True,
copyExtras = True,
chi2n_par = 0.9,
res_par = ( 0.003, 0.001 ),
minNumberLayers = 3,
##minNumber3DLayers = 3,
minNumber3DLayers = 1,
maxNumberLostLayers = 1,
d0_par1 = ( 0.9, 3.0 ),
dz_par1 = ( 0.9, 3.0 ),
d0_par2 = ( 1.0, 3.0 ),
dz_par2 = ( 1.0, 3.0 )
)

thStepTrk = RecoTracker.FinalTrackSelectors.selectHighPurity_cfi.selectHighPurity.clone(
src = 'iterativeThirdTrackMerging',
copyTrajectories = True,
chi2n_par = 0.5,
res_par = ( 0.003, 0.001 ),
#minNumberLayers = 5,
minNumberLayers = 3,
#minNumber3DLayers = 4,
minNumber3DLayers = 1,
maxNumberLostLayers = 1,
d0_par1 = ( 1.0, 4.0 ),
dz_par1 = ( 1.0, 4.0 ),
d0_par2 = ( 1.0, 4.0 ),
dz_par2 = ( 1.0, 4.0 )
)

##import RecoTracker.FinalTrackSelectors.ctfrsTrackListMerger_cfi
##thStep = RecoTracker.FinalTrackSelectors.ctfrsTrackListMerger_cfi.ctfrsTrackListMerger.clone()
##thStep.TrackProducer1 = 'thStepVtx'
##thStep.TrackProducer2 = 'thStepTrk'


thStep = cms.EDProducer("FastTrackMerger",
                      TrackProducers = cms.VInputTag(cms.InputTag("thStepVtx"),
                                                     cms.InputTag("thStepTrk"))
)

thfilter = cms.EDProducer("QualityFilter",
    TrackQuality = cms.string('highPurity'),
    recTracks = cms.InputTag("thStep")
)

iterativeThirdTrackFiltering = cms.Sequence(thStepVtx*thStepTrk*thStep*thfilter)
