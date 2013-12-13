
import FWCore.ParameterSet.Config as cms

HCALHighEnergyFilter = cms.EDFilter("HCALHighEnergyFilter",
                                    JetTag = cms.InputTag("ak4CaloJets"),
                                    JetThreshold = cms.double(20),
                                    EtaCut = cms.double(3.0)
#                                    CentralJets  = cms.untracked.InputTag("hltL1extraParticles","Central"),
#                                    TauJets  = cms.untracked.InputTag("hltL1extraParticles","Tau")
                                    )
