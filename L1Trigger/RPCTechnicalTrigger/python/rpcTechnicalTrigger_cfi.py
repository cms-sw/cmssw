import FWCore.ParameterSet.Config as cms

#.............................................................................
# Default configurations

rpcTechnicalTrigger  = cms.EDProducer('RPCTechnicalTrigger',
                                      RPCDigiLabel = cms.InputTag("simMuonRPCDigis"),
                                      RPCSimLinkInstance = cms.string("RPCDigiSimLink"),
                                      UseRPCSimLink = cms.untracked.int32(0),
                                      UseEventSetup = cms.untracked.int32(0),
                                      BitNumbers=cms.vuint32(24,25,26,27,28,29,30),
                                      BitNames=cms.vstring('L1Tech_RPC_TTU_barrel_Cosmics/v0',
                                                           'L1Tech_RPC_TTU_pointing_Cosmics/v0',
                                                           'L1Tech_RPC_TTU_RBplus2_Cosmics/v0',
                                                           'L1Tech_RPC_TTU_RBplus1_Cosmics/v0',
                                                           'L1Tech_RPC_TTU_RB0_Cosmics/v0',
                                                           'L1Tech_RPC_TTU_RBminus1_Cosmics/v0',
                                                           'L1Tech_RPC_TTU_RBminus2_Cosmics/v0',
                                                           ) )

