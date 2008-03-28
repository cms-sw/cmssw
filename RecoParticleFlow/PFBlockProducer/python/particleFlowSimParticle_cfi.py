import FWCore.ParameterSet.Config as cms

particleFlowSimParticle = cms.EDProducer("PFSimParticleProducer",
    # verbosity 
    verbose = cms.untracked.bool(False),
    # Tracking parameters
    Fitter = cms.string('KFFittingSmoother'),
    # replace ParticleFilter.pTMin = 0.5
    # flags 
    process_RecTracks = cms.untracked.bool(False),
    ParticleFilter = cms.PSet(
        EProton = cms.double(5000.0),
        # Particles with |eta| > etaMax (momentum direction at primary vertex) 
        # are not simulated 
        etaMax = cms.double(5.0),
        # Charged particles with pT < pTMin (GeV/c) are not simulated
        pTMin = cms.double(0.0),
        # Particles with energy smaller than EMin (GeV) are not simulated
        EMin = cms.double(0.0)
    ),
    TTRHBuilder = cms.string('WithTrackAngle'),
    # input collections ----------------------------------------
    # module label to find input sim tracks and sim vertices
    sim = cms.InputTag("g4SimHits"),
    process_Particles = cms.untracked.bool(True),
    Propagator = cms.string('PropagatorWithMaterial'),
    VertexGenerator = cms.PSet(
        type = cms.string('None')
    )
)


