import FWCore.ParameterSet.Config as cms
from HLTrigger.Configuration.HLT_GRun_Famos_cff import *

# remove the calorechit producers from the HLT sequences                                                                                                                                                   
_toremove = [hltEcalRegionalPi0RecHit,hltHfreco,hltHoreco,hltHbhereco,hltEcalRecHit,hltEcalPreshowerRecHit]
for _key,_value in locals().items():
    if isinstance(_value,cms.Sequence):
        for _entry in _toremove:
            _value.remove(_entry)

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


hltEcalRecHit.killDeadChannels = False
hltEcalRecHit.recoverEBFE = False
hltEcalRecHit.recoverEEFE = False


# remove digi producers from the HLT sequences                                                                                                                                                   
_toremove = [hltEcalPreshowerDigis,hltEcalDigis,hltEcalDetIdToBeRecovered ]
for _key,_value in locals().items():
    if isinstance(_value,cms.Sequence):
        for _entry in _toremove:
            _value.remove(_entry)

import FastSimulation.Configuration.DigiAndMixAliasInfo_cff as _aliasInfo
hltEcalPreshowerDigis = _aliasInfo.infoToAlias(_aliasInfo.ecalPreShowerDigisAliasInfo)
hltEcalDigis = _aliasInfo.infoToAlias(_aliasInfo.ecalDigisAliasInfo)
