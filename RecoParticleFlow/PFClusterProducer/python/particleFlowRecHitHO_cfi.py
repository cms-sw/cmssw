import FWCore.ParameterSet.Config as cms

particleFlowRecHitHO = cms.EDProducer("PFRecHitProducerHO",
    # verbosity 
    verbose = cms.untracked.bool(False),
    # The collection of HO rechits
    recHitsHO = cms.InputTag("horeco", ""), # for RECO
    # The threshold for rechit energies in ring0
    thresh_Barrel = cms.double(0.4),
    # The threshold for rechit energies in rings +/-1 and +/-2
    thresh_Endcap = cms.double(1.0),
                                      
)


