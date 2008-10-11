import FWCore.ParameterSet.Config as cms

particleFlow = cms.EDProducer("PFProducer",
    pf_mergedPhotons_mvaWeightFile = cms.string('RecoParticleFlow/PFProducer/data/MVAnalysis_MLP.weights.txt'),
    # Tracking parameters
    #string Fitter      = "KFFittingSmoother"   
    #string Propagator  = "PropagatorWithMaterial" 
    #string TTRHBuilder = "WithTrackAngle"
    # input collections ----------------------------------------
    blocks = cms.InputTag("particleFlowBlock"),
    # debugging         ----------------------------------------
    # verbosity 
    verbose = cms.untracked.bool(False),
    #Alogrithm to recover the HCAL clusters (produced
    #by nuclear interactions) that belong to charged hadrons
    pf_clusterRecovery = cms.bool(False),
    pf_calib_ECAL_HCAL_eslope = cms.double(1.05),
    pf_mergedPhotons_mvaCut = cms.double(0.5),
    pf_calib_HCAL_offset = cms.double(1.73),
    pf_calib_HCAL_slope = cms.double(2.17),
    pf_calib_HCAL_damping = cms.double(2.49),
    # particle flow parameters ---------------------------------
    # number of sigma for creating neutral particles
    pf_nsigma_ECAL = cms.double(3.0),
    pf_calib_ECAL_HCAL_hslope = cms.double(1.06),
    # algo type (0: main, 1:electrons, other not yet implemented)
    algoType = cms.uint32(0),
    pf_calib_ECAL_HCAL_offset = cms.double(6.11),
    pf_mergedPhotons_PSCut = cms.double(0.001),
    pf_calib_ECAL_slope = cms.double(1.0),
    # debugging
    debug = cms.untracked.bool(False),
    pf_calib_ECAL_offset = cms.double(0.0),
    pf_nsigma_HCAL = cms.double(1.7)
)


