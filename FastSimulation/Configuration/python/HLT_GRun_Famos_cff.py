import FWCore.ParameterSet.Config as cms
from HLTrigger.Configuration.HLT_GRun_Famos_cff import *

# remove the calorechit producers from the HLT sequences                                                                                                                                                   
_toremove = [fragment.hltHfreco,fragment.hltHoreco,fragment.hltHbhereco]
for _seq in fragment.sequences_().values():
    for _entry in _toremove:
        _seq.remove(_entry)

fragment.hltHbhereco = cms.EDAlias(
    hbhereco = cms.VPSet(
        cms.PSet(
            type = cms.string("HBHERecHitsSorted")
            )
        )
    )


fragment.hltHoreco = cms.EDAlias(
    horeco = cms.VPSet(
        cms.PSet(
            type = cms.string("HORecHitsSorted")
            )
        )
    )

fragment.hltHfreco = cms.EDAlias(
    hfreco = cms.VPSet(
        cms.PSet(
            type = cms.string("HFRecHitsSorted")
            )
        )
    )

fragment.hltEcalRecHit.killDeadChannels = False
fragment.hltEcalRecHit.recoverEBFE = False
fragment.hltEcalRecHit.recoverEEFE = False

# remove digi producers from the HLT sequences                                                                                                                                                   
_toremove = [fragment.hltEcalPreshowerDigis,fragment.hltEcalDigis,fragment.hltEcalDetIdToBeRecovered ]
for _seq in fragment.sequences_().values():
    for _entry in _toremove:
        _seq.remove(_entry)

import FastSimulation.Configuration.DigiAndMixAliasInfo_cff as _aliasInfo
fragment.hltEcalPreshowerDigis = _aliasInfo.infoToAlias(_aliasInfo.ecalPreShowerDigisAliasInfo)
fragment.hltEcalDigis = _aliasInfo.infoToAlias(_aliasInfo.ecalDigisAliasInfo)

for _entry in [fragment.HLT_MET75_IsoTrk50_v1,fragment.HLT_MET90_IsoTrk50_v1]:
    fragment.HLTSchedule.remove(_entry)
