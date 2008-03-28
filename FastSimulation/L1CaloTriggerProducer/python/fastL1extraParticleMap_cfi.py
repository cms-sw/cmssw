import FWCore.ParameterSet.Config as cms

import copy
from L1Trigger.L1ExtraFromDigis.l1extraParticleMap_cfi import *
fastL1extraParticleMap = copy.deepcopy(l1extraParticleMap)
fastL1extraParticleMap.isolatedEmSource = cms.InputTag("fastL1CaloSim","Isolated")
fastL1extraParticleMap.nonIsolatedEmSource = cms.InputTag("fastL1CaloSim","NonIsolated")
fastL1extraParticleMap.centralJetSource = cms.InputTag("fastL1CaloSim","Central")
fastL1extraParticleMap.forwardJetSource = cms.InputTag("fastL1CaloSim","Forward")
fastL1extraParticleMap.tauJetSource = cms.InputTag("fastL1CaloSim","Tau")
fastL1extraParticleMap.muonSource = 'fastL1CaloSim'
fastL1extraParticleMap.etMissSource = 'fastL1CaloSim'

