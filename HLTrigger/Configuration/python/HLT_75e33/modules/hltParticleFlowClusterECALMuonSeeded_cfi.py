import FWCore.ParameterSet.Config as cms

from .hltParticleFlowClusterECALUnseeded_cfi import hltParticleFlowClusterECALUnseeded as _hltParticleFlowClusterECALUnseeded

hltParticleFlowClusterECALMuonSeeded = _hltParticleFlowClusterECALUnseeded.clone(
    energyCorrector = dict(
        recHitsEBLabel = "hltEcalRecHitMuonSeeded:EcalRecHitsEB",
        recHitsEELabel = "hltEcalRecHitMuonSeeded:EcalRecHitsEE"
    ),
    inputECAL = "hltParticleFlowClusterECALUncorrectedMuonSeeded"
)
