import FWCore.ParameterSet.Config as cms

import  L1Trigger.L1ExtraFromDigis.l1extraParticleMap_cfi 
fastL1extraParticleMap = L1Trigger.L1ExtraFromDigis.l1extraParticleMap_cfi.l1extraParticleMap.clone()
fastL1extraParticleMap.isolatedEmSource = cms.InputTag("fastL1CaloSim","Isolated")
fastL1extraParticleMap.nonIsolatedEmSource = cms.InputTag("fastL1CaloSim","NonIsolated")
fastL1extraParticleMap.centralJetSource = cms.InputTag("fastL1CaloSim","Central")
fastL1extraParticleMap.forwardJetSource = cms.InputTag("fastL1CaloSim","Forward")
fastL1extraParticleMap.tauJetSource = cms.InputTag("fastL1CaloSim","Tau")
fastL1extraParticleMap.muonSource = cms.InputTag("fastL1CaloSim")
fastL1extraParticleMap.etMissSource = cms.InputTag("fastL1CaloSim","MET")
fastL1extraParticleMap.htMissSource = cms.InputTag("fastL1CaloSim","MET") # MHT not supported yet!
#fastL1extraParticleMap.etMissSource = cms.InputTag("fastL1CaloSim","MET")
#fastL1extraParticleMap.htMissSource = cms.InputTag("fastL1CaloSim","MHT")

