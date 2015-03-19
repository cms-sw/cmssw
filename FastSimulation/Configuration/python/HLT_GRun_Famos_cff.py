import FWCore.ParameterSet.Config as cms
from HLTrigger.Configuration.HLT_GRun_Famos_cff import *

# remove track producers from trigger path
_toremove = [
    HLTIterativeTrackingIter04,
    HLTIterativeTrackingIter02,
    HLTIterativeTracking,
    HLTIterativeTrackingTau3Mu,
    HLTIterativeTrackingReg,
    HLTIterativeTrackingForElectronIter02,
    HLTIterativeTrackingForPhotonsIter02,
    HLTIterativeTrackingL3MuonIter02,
    HLTIterativeTrackingL3MuonRegIter02,
    HLTIterativeTrackingForBTagIter02,
    HLTIterativeTrackingForTauIter02,
    HLTIterativeTrackingForTauIter04,
    HLTIterativeTrackingGlbTrkMuonIter02,
    HLTIterativeTrackingHighPtTkMu,
    HLTIterativeTrackingHighPtTkMuIsoIter02,
    HLTIterativeTrackingDisplacedJpsiIter02,
    HLTIterativeTrackingDisplacedPsiPrimeIter02,
    HLTIterativeTrackingDisplacedNRMuMuIter02,
    HLTIterativeTrackingForBTagIter12,
    HLTIterativeTrackingForBTagIteration0,
    HLTIterativeTrackingIteration4DisplacedJets
    ]
for _key,_value in locals().items():
    if isinstance(_value,cms.Path):
        for _entry in _toremove:
            _value.remove(_entry)

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
