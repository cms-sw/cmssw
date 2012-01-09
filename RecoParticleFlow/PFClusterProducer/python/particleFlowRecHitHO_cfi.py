import FWCore.ParameterSet.Config as cms

particleFlowRecHitHO = cms.EDProducer("PFRecHitProducerHO",
    # verbosity 
    verbose = cms.untracked.bool(False),
    recHitsHO = cms.InputTag("horeco", ""), # for RECO
#     recHitsHO = cms.InputTag("reducedHcalRecHits"        "hfreco"), # for AOD
    threshold_R0 = cms.double(0.4),
    threshold_R1 = cms.double(1.0),
                                      
)


