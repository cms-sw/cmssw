import FWCore.ParameterSet.Config as cms

from ..modules.hltBunchSpacingProducer_cfi import *
from ..modules.hltEcalBarrelDigisMuonSeeded_cfi import *
from ..modules.hltEcalDetIdToBeRecovered_cfi import *
from ..modules.hltEcalDigis_cfi import *
from ..modules.hltEcalRecHitMuonSeeded_cfi import *
from ..modules.hltEcalUncalibRecHitMuonSeeded_cfi import *
from ..modules.hltParticleFlowClusterECALMuonSeeded_cfi import *
from ..modules.hltParticleFlowClusterECALUncorrectedMuonSeeded_cfi import *
from ..modules.hltParticleFlowRecHitECALMuonSeeded_cfi import *

HLTEcalPFClusteringMuonSeededSequence = cms.Sequence(
    hltEcalDigis
    + hltBunchSpacingProducer
    + hltEcalDetIdToBeRecovered
    + hltEcalBarrelDigisMuonSeeded
    + hltEcalUncalibRecHitMuonSeeded
    + hltEcalRecHitMuonSeeded
    + hltParticleFlowRecHitECALMuonSeeded
    + hltParticleFlowClusterECALUncorrectedMuonSeeded
    + hltParticleFlowClusterECALMuonSeeded
)
