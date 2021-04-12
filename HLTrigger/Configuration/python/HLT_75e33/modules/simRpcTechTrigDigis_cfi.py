import FWCore.ParameterSet.Config as cms

simRpcTechTrigDigis = cms.EDProducer("RPCTechnicalTrigger",
    BitNames = cms.vstring(
        'L1Tech_RPC_TTU_barrel_Cosmics/v0',
        'L1Tech_RPC_TTU_pointing_Cosmics/v0',
        'L1Tech_RPC_TTU_RBplus2_Cosmics/v0',
        'L1Tech_RPC_TTU_RBplus1_Cosmics/v0',
        'L1Tech_RPC_TTU_RB0_Cosmics/v0',
        'L1Tech_RPC_TTU_RBminus1_Cosmics/v0',
        'L1Tech_RPC_TTU_RBminus2_Cosmics/v0'
    ),
    BitNumbers = cms.vuint32(
        24, 25, 26, 27, 28,
        29, 30
    ),
    ConfigFile = cms.string('hardware-pseudoconfig.txt'),
    RPCDigiLabel = cms.InputTag("simMuonRPCDigis"),
    RPCSimLinkInstance = cms.InputTag("RPCDigiSimLink"),
    UseEventSetup = cms.untracked.int32(0),
    UseRPCSimLink = cms.untracked.int32(0),
    Verbosity = cms.untracked.int32(0)
)
