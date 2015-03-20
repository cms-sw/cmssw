import FWCore.ParameterSet.Config as cms
from HLTrigger.Configuration.HLT_GRun_Famos_cff import *
            
hltEcalPreshowerRecHit = cms.EDAlias(
    ecalPreshowerRecHit = cms.VPSet(
        cms.PSet(
            type = cms.string("EcalRecHitsSorted"),
            fromProductInstance = cms.string("EcalRecHitsES")
            )
        )
    )


hltEcalRecHit = cms.EDAlias(
    ecalRecHit = cms.VPSet(
        cms.PSet(
            type = cms.string("EcalRecHitsSorted"),
            fromProductInstance = cms.string("EcalRecHitsEB")),
        cms.PSet(
            type = cms.string("EcalRecHitsSorted"),
            fromProductInstance = cms.string("EcalRecHitsEE")),
        )
    )

hltHbhereco = cms.EDAlias(
    hbhereco = cms.VPSet(
        cms.PSet(
            type = cms.string("HBHERecHitsSorted")
            )
        )
    )


hltHoreco = cms.EDAlias(
    horeco = cms.VPSet(
        cms.PSet(
            type = cms.string("HORecHitsSorted")
            )
        )
    )

hltHfreco = cms.EDAlias(
    hfreco = cms.VPSet(
        cms.PSet(
            type = cms.string("HFRecHitsSorted")
            )
        )
    )

hltEcalRegionalPi0RecHit = cms.EDAlias(ecalRecHit = hltEcalRecHit.ecalRecHit)
