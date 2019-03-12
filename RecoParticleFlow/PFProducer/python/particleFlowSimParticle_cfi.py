import FWCore.ParameterSet.Config as cms

from FastSimulation.Event.ParticleFilter_cfi import ParticleFilterBlock

particleFlowSimParticle = cms.EDProducer("PFSimParticleProducer",
    # verbosity 
    verbose = cms.untracked.bool(False),
    # Tracking parameters
    Fitter = cms.string('KFFittingSmoother'),
    # replace ParticleFilter.pTMin = 0.5
    # flags 
    process_RecTracks = cms.untracked.bool(False),
    #
    ParticleFilter = ParticleFilterBlock.ParticleFilter.clone(chargedPtMin = 0, EMin = 0),
    #
    TTRHBuilder = cms.string('WithTrackAngle'),
    process_Particles = cms.untracked.bool(True),
    Propagator = cms.string('PropagatorWithMaterial'),
    # input collections ----------------------------------------
    # module label to find input sim tracks and sim vertices
    sim = cms.InputTag("g4SimHits"),
    #Monte Carlo Truth Matching Options, only in FASTSIM!:
    #MC Truth Matching info (only if UnFoldedMode = true in FastSim) 
    MCTruthMatchingInfo = cms.untracked.bool(False), 
    #retrieving RecTracks
    RecTracks = cms.InputTag("trackerDrivenElectronSeeds"),                                 
    #retrieving EcalRechits
    ecalRecHitsEB = cms.InputTag('caloRecHits','EcalRecHitsEB'),
    ecalRecHitsEE = cms.InputTag('caloRecHits','EcalRecHitsEE'),
    #retrieving fastSim SimHits                                     
    fastSimProducer = cms.untracked.InputTag('fastSimProducer','EcalHitsEB')
)

