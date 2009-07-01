
import FWCore.ParameterSet.Config as cms

HCALHighEnergyFilter = cms.EDFilter("HCALHighEnergyFilter",
                             JetThreshold = cms.untracked.double(20),
                             CentralJets  = cms.untracked.InputTag("hltL1extraParticles","Central"),
                             TauJets  = cms.untracked.InputTag("hltL1extraParticles","Tau")
                             )
