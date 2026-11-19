import FWCore.ParameterSet.Config as cms

from .hltParticleFlowRecHitECALUnseeded_cfi import hltParticleFlowRecHitECALUnseeded as _hltParticleFlowRecHitECALUnseeded

hltParticleFlowRecHitECALMuonSeeded = _hltParticleFlowRecHitECALUnseeded.clone(
    producers = {
        0: dict(src = "hltEcalRecHitMuonSeeded:EcalRecHitsEB"),
        1: dict(src = "hltEcalRecHitMuonSeeded:EcalRecHitsEE")
    }
)
