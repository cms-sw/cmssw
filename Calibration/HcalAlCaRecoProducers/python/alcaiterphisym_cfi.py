import FWCore.ParameterSet.Config as cms

# producer for alcaiterativephisym (HCAL Iterative Phi Symmetry)


IterativePhiSymProd = cms.EDProducer("AlCaEcalHcalReadoutsProducer",
    hbheInput = cms.InputTag("hbhereco"),
    hoInput = cms.InputTag("horeco"),
    hfInput = cms.InputTag("hfreco")
)
