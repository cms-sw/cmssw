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

    # Maximum allowed severity of HO rechits.  Hits above the given severity level will be rejected.  Default max value is 11 (the same value as used for allowing hits in PF caloTowers, and the expected acceptance value of any PF HCAL hits)
    HOMaxAllowedSev = cms.int32(11),
                                        
)


