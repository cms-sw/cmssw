import FWCore.ParameterSet.Config as cms

hbheplan1 = cms.EDProducer(
    "HBHEPlan1Combiner",

    # Label for the input HBHERecHitCollection
    hbheInput = cms.InputTag("hbheprereco"),

    # Should we ignore "Plan 1" settings provided by the HcalTopology class?
    ignorePlan1Topology = cms.bool(False),

    # If we are ignoring HcalTopology, should the rechits be combined
    # according to the "Plan 1" scheme? This flag is meaningful only if
    # "ignorePlan1Topology" is True.
    usePlan1Mode = cms.bool(True),

    # Configure the rechit combination algorithm
    algorithm = cms.PSet(
        Class = cms.string("SimplePlan1RechitCombiner")
    )
)
