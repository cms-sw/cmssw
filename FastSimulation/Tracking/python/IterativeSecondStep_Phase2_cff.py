import FWCore.ParameterSet.Config as cms

### STEP 0 ###


# seeding
import FastSimulation.Tracking.TrajectorySeedProducer_cfi
iterativeSecondSeeds = FastSimulation.Tracking.TrajectorySeedProducer_cfi.trajectorySeedProducer.clone()
iterativeSecondSeeds.firstHitSubDetectorNumber = [1]
iterativeSecondSeeds.firstHitSubDetectors = [2]
iterativeSecondSeeds.secondHitSubDetectorNumber = [1]
iterativeSecondSeeds.secondHitSubDetectors = [ 2]
iterativeSecondSeeds.thirdHitSubDetectorNumber = [1]
iterativeSecondSeeds.thirdHitSubDetectors = [2]
iterativeSecondSeeds.seedingAlgo = ['SecondPixelTriplets']
iterativeSecondSeeds.minRecHits = [3] 
iterativeSecondSeeds.pTMin = [0.3]
iterativeSecondSeeds.maxD0 = [1.]
iterativeSecondSeeds.maxZ0 = [30.]
iterativeSecondSeeds.numberOfHits = [3]
iterativeSecondSeeds.originRadius = [1.0] # note: standard tracking uses 0.03, but this value gives a much better agreement in rate and shape for iter0
iterativeSecondSeeds.originHalfLength = [15.9] 
iterativeSecondSeeds.originpTMin = [0.6] 
iterativeSecondSeeds.zVertexConstraint = [-1.0]
iterativeSecondSeeds.primaryVertices = ['none']
# new (AG)
iterativeSecondSeeds.newSyntax = False
#new PA from upgrade code
iterativeSecondSeeds.layerList = ['treyrey+hgkg879+ht7n9']


# candidate producer
import FastSimulation.Tracking.TrackCandidateProducer_cfi
iterativeSecondTrackCandidates = FastSimulation.Tracking.TrackCandidateProducer_cfi.trackCandidateProducer.clone()
iterativeSecondTrackCandidates.SeedProducer = cms.InputTag("iterativeSecondSeeds","SecondPixelTriplets")
iterativeSecondTrackCandidates.TrackProducers = ['globalPixelWithMaterialTracks']
iterativeSecondTrackCandidates.MinNumberOfCrossedLayers = 3

# track producer
import RecoTracker.TrackProducer.CTFFinalFitWithMaterial_cfi
iterativeSecondTracks = RecoTracker.TrackProducer.CTFFinalFitWithMaterial_cfi.ctfWithMaterialTracks.clone()
iterativeSecondTracks.src = 'iterativeSecondTrackCandidates'
iterativeSecondTracks.TTRHBuilder = 'WithoutRefit'
iterativeSecondTracks.Fitter = 'KFFittingSmootherWithOutlierRejection'
iterativeSecondTracks.Propagator = 'PropagatorWithMaterial'

# track merger
secondStepTracks = cms.EDProducer("FastTrackMerger",
                                   TrackProducers = cms.VInputTag(cms.InputTag("iterativeSecondTrackCandidates"),
                                                                  cms.InputTag("iterativeSecondTracks")),
                                   trackAlgo = cms.untracked.uint32(4) # iter0
                                   )

# Final selection
import RecoTracker.FinalTrackSelectors.multiTrackSelector_cfi

secondStepSelector = RecoTracker.FinalTrackSelectors.multiTrackSelector_cfi.multiTrackSelector.clone(
    src='secondStepTracks',
    trackSelectors= cms.VPSet(
    RecoTracker.FinalTrackSelectors.multiTrackSelector_cfi.looseMTS.clone(
    name = 'secondStepLoose',
    vertexCut = cms.string(''),
    # Boolean indicating if adapted primary vertex compatibility cuts are to be applied.
    applyAdaptedPVCuts = cms.bool(False),

    # from here on it is default (can even be removed)
    # vertex selection 
    vtxNumber = cms.int32(-1),
    
    #untracked bool copyTrajectories = true // when doing retracking before
    copyTrajectories = cms.untracked.bool(False),
    copyExtras = cms.untracked.bool(True), ## set to false on AOD
    qualityBit = cms.string('loose'), ## set to '' or comment out if you dont want to set the bit
    
    # parameters for adapted optimal cuts on chi2 and primary vertex compatibility
    chi2n_par = cms.double(1.6),
    chi2n_no1Dmod_par = cms.double(9999),
    res_par = cms.vdouble(0.003, 0.01),
    d0_par1 = cms.vdouble(0.55, 4.0),
    dz_par1 = cms.vdouble(0.65, 4.0),
    d0_par2 = cms.vdouble(0.55, 4.0),
    dz_par2 = cms.vdouble(0.45, 4.0),
    
    # Impact parameter absolute cuts.
    max_d0 = cms.double(100.),
    max_z0 = cms.double(100.),
    nSigmaZ = cms.double(4.),
    
    # Cuts on numbers of layers with hits/3D hits/lost hits. 
    minNumberLayers = cms.uint32(0),
    minNumber3DLayers = cms.uint32(0),
    maxNumberLostLayers = cms.uint32(999),
    minHitsToBypassChecks = cms.uint32(20),

    # Absolute cuts in case of no PV. If yes, please define also max_d0NoPV and max_z0NoPV 
    applyAbsCutsIfNoPV = cms.bool(False),
    keepAllTracks= cms.bool(False),

    # parameters for cutting on pterror/pt and number of valid hits
    max_relpterr = cms.double(9999.),
    min_nhits = cms.uint32(0),

    max_minMissHitOutOrIn = cms.int32(99),
    max_lostHitFraction = cms.double(1.0),

    # parameters for cutting on eta
    min_eta = cms.double(-9999.),
    max_eta = cms.double(9999.),


    ), #end of pset
    RecoTracker.FinalTrackSelectors.multiTrackSelector_cfi.tightMTS.clone(
    name = 'secondStepTight',
    preFilterName = 'secondStepLoose',
    minNumberLayers = cms.uint32(0),
    minNumber3DLayers = cms.uint32(0),
    maxNumberLostLayers = cms.uint32(999),
    vertexCut = cms.string(''),

    # from here on it is default (can even be removed)
    d0_par1 = cms.vdouble(0.3, 4.0),
    dz_par1 = cms.vdouble(0.35,4.0),
    d0_par2 = cms.vdouble(0.4, 4.0),
    dz_par2 = cms.vdouble(0.4, 4.0),
    chi2n_par = cms.double(0.7),
    chi2n_no1Dmod_par = cms.double(9999),
    qualityBit = cms.string('tight'), ## set to '' or comment out if you dont want to set the bit
    keepAllTracks= cms.bool(True),

    ),
    RecoTracker.FinalTrackSelectors.multiTrackSelector_cfi.highpurityMTS.clone(
    name = 'secondStep',
    preFilterName = 'secondStepTight',
    minNumberLayers = cms.uint32(0),
    minNumber3DLayers = cms.uint32(0),
    maxNumberLostLayers = cms.uint32(999),
    vertexCut = cms.string(''),
    # from here on it is default (can even be removed)
    res_par=cms.vdouble(0.003,0.001),
    qualityBit = cms.string('highPurity'), ## set to '' or comment out if you dont want to set the bit
    ),
    ) #end of vpset
    ) #end of clone


# Final sequence
iterativeSecondStep = cms.Sequence(iterativeSecondSeeds
                                    +iterativeSecondTrackCandidates
                                    +iterativeSecondTracks
                                    +secondStepTracks
                                    +secondStepSelector)



