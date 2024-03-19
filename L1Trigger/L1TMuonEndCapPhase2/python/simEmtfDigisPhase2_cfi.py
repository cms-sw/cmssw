import FWCore.ParameterSet.Config as cms
from Configuration.Eras.Modifier_phase2_GE0_cff import phase2_GE0

# EMTF Phase2 emulator configuration
simEmtfDigisMCPhase2 = cms.EDProducer(
    "L1TMuonEndCapPhase2TrackProducer",

    # Verbosity level
    Verbosity = cms.untracked.int32(3),

    # Validation
    ValidationDirectory = cms.string("L1Trigger/L1TMuonEndCapPhase2/data/validation"),

    # Neural Network Models
    PromptGraphPath = cms.string("L1Trigger/L1TMuonEndCapPhase2/data/prompt_model.pb"),
    DisplacedGraphPath = cms.string("L1Trigger/L1TMuonEndCapPhase2/data/displaced_model.pb"),

    # Input collections
    # Three options for CSCInput
    #   * 'simCscTriggerPrimitiveDigis','MPCSORTED' : simulated trigger primitives (LCTs) from re-emulating CSC digis
    #   * 'csctfDigis' : real trigger primitives as received by CSCTF (legacy trigger), available only in 2016 data
    #   * 'emtfStage2Digis' : real trigger primitives as received by EMTF, unpacked in EventFilter/L1TRawToDigi/
    CSCInput = cms.InputTag('simCscTriggerPrimitiveDigisForEMTF','MPCSORTED'),
    RPCInput = cms.InputTag('rpcRecHitsForEMTF'),
    GEMInput = cms.InputTag('simMuonGEMPadDigiClusters'),
    ME0Input = cms.InputTag('me0TriggerConvertedPseudoDigis'),
    GE0Input = cms.InputTag('ge0TriggerConvertedPseudoDigis'),

    # Run with CSC, RPC, GEM
    CSCEnabled = cms.bool(True),  # Use CSC LCTs from the MPCs in track-building
    RPCEnabled = cms.bool(True),  # Use clustered RPC hits from CPPF in track-building
    GEMEnabled = cms.bool(True),  # Use hits from GEMs in track-building
    ME0Enabled = cms.bool(True),
    GE0Enabled = cms.bool(False),

    # BX
    MinBX    = cms.int32(-2), # Minimum BX considered
    MaxBX    = cms.int32(2), # Maximum BX considered
    BXWindow = cms.int32(1),  # Number of BX whose primitives can be included in the same track

    CSCInputBXShift = cms.int32(-8), # Shift applied to input CSC LCT primitives, to center at BX = 0
    RPCInputBXShift = cms.int32(0),
    GEMInputBXShift = cms.int32(0),
    ME0InputBXShift = cms.int32(-8),

    IncludeNeighborEnabled = cms.bool(True),  # Include primitives from neighbor chambers in track-building
)

phase2_GE0.toModify(simEmtfDigisMCPhase2, ME0Enabled=False, GE0Enabled=True)

simEmtfDigisDataPhase2 = simEmtfDigisMCPhase2.clone(
    # Inputs
    CSCInput = cms.InputTag('emtfStage2Digis'),
    RPCInput = cms.InputTag('muonRPCDigis'),
)

simEmtfDigisPhase2 = simEmtfDigisMCPhase2.clone()
