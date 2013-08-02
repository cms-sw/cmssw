import FWCore.ParameterSet.Config as cms

### STEP 0 ###


# seeding
import FastSimulation.Tracking.TrajectorySeedProducer_cfi
iterativeInitialSeeds = FastSimulation.Tracking.TrajectorySeedProducer_cfi.trajectorySeedProducer.clone()
iterativeInitialSeeds.firstHitSubDetectorNumber = [2]
iterativeInitialSeeds.firstHitSubDetectors = [1,2]
iterativeInitialSeeds.secondHitSubDetectorNumber = [2]
iterativeInitialSeeds.secondHitSubDetectors = [1, 2]
iterativeInitialSeeds.thirdHitSubDetectorNumber = [2]
iterativeInitialSeeds.thirdHitSubDetectors = [1, 2]
iterativeInitialSeeds.seedingAlgo = ['InitialPixelTriplets']
iterativeInitialSeeds.minRecHits = [3] 
iterativeInitialSeeds.pTMin = [0.3]
iterativeInitialSeeds.maxD0 = [1.]
iterativeInitialSeeds.maxZ0 = [30.]
iterativeInitialSeeds.numberOfHits = [3]
iterativeInitialSeeds.originRadius = [1.0] # note: standard tracking uses 0.03, but this value gives a much better agreement in rate and shape for iter0
iterativeInitialSeeds.originHalfLength = [15.9] 
iterativeInitialSeeds.originpTMin = [0.6] 
iterativeInitialSeeds.zVertexConstraint = [-1.0]
iterativeInitialSeeds.primaryVertices = ['none']
# new (AG)
iterativeInitialSeeds.newSyntax = False
#new PA from upgrade code
iterativeInitialSeeds.layerList = ['BPix1+BPix2+BPix3',
                                           'BPix2+BPix3+BPix4',
                                             'BPix1+BPix3+BPix4',
                                             'BPix1+BPix2+BPix4',
                                             'BPix2+BPix3+FPix1_pos', 'BPix2+BPix3+FPix1_neg',
                                             'BPix1+BPix2+FPix1_pos', 'BPix1+BPix2+FPix1_neg',
                                             'BPix1+BPix3+FPix1_pos', 'BPix1+BPix3+FPix1_neg',
                                             'BPix2+FPix1_pos+FPix2_pos', 'BPix2+FPix1_neg+FPix2_neg',
                                             'BPix1+FPix1_pos+FPix2_pos', 'BPix1+FPix1_neg+FPix2_neg',
                                             'BPix1+BPix2+FPix2_pos', 'BPix1+BPix2+FPix2_neg',
                                             'FPix1_pos+FPix2_pos+FPix3_pos', 'FPix1_neg+FPix2_neg+FPix3_neg',
                                             'BPix1+FPix2_pos+FPix3_pos', 'BPix1+FPix2_neg+FPix3_neg',
                                             'BPix1+FPix1_pos+FPix3_pos', 
					     'BPix1+FPix1_neg+FPix3_neg']


# candidate producer
import FastSimulation.Tracking.TrackCandidateProducer_cfi
iterativeInitialTrackCandidates = FastSimulation.Tracking.TrackCandidateProducer_cfi.trackCandidateProducer.clone()
iterativeInitialTrackCandidates.SeedProducer = cms.InputTag("iterativeInitialSeeds","InitialPixelTriplets")
iterativeInitialTrackCandidates.TrackProducers = ['globalPixelWithMaterialTracks']
iterativeInitialTrackCandidates.MinNumberOfCrossedLayers = 3

# track producer
import RecoTracker.TrackProducer.CTFFinalFitWithMaterial_cfi
iterativeInitialTracks = RecoTracker.TrackProducer.CTFFinalFitWithMaterial_cfi.ctfWithMaterialTracks.clone()
iterativeInitialTracks.src = 'iterativeInitialTrackCandidates'
iterativeInitialTracks.TTRHBuilder = 'WithoutRefit'
iterativeInitialTracks.Fitter = 'KFFittingSmootherWithOutlierRejection'
iterativeInitialTracks.Propagator = 'PropagatorWithMaterial'

# track merger
initialStepTracks = cms.EDProducer("FastTrackMerger",
                                   TrackProducers = cms.VInputTag(cms.InputTag("iterativeInitialTrackCandidates"),
                                                                  cms.InputTag("iterativeInitialTracks")),
                                   trackAlgo = cms.untracked.uint32(4) # iter0
                                   )

# Final selection
import RecoTracker.FinalTrackSelectors.multiTrackSelector_cfi

initialStepSelector = RecoTracker.FinalTrackSelectors.multiTrackSelector_cfi.multiTrackSelector.clone(
    src='initialStepTracks',
    trackSelectors= cms.VPSet(
    RecoTracker.FinalTrackSelectors.multiTrackSelector_cfi.looseMTS.clone(
    name = 'initialStepLoose',
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
    name = 'initialStepTight',
    preFilterName = 'initialStepLoose',
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
    name = 'initialStep',
    preFilterName = 'initialStepTight',
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
iterativeInitialStep = cms.Sequence(iterativeInitialSeeds
                                    +iterativeInitialTrackCandidates
                                    +iterativeInitialTracks
                                    +initialStepTracks
                                    +initialStepSelector)



