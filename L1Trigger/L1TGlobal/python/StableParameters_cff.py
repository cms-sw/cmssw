#
# WARNING: This file is in the L1T configuration critical path.
#
# All changes must be explicitly discussed with the L1T offline coordinator.
#
import FWCore.ParameterSet.Config as cms

StableParametersRcdSource = cms.ESSource("EmptyESSource",
    recordName = cms.string('L1TGlobalStableParametersRcd'),
    iovIsRunNotTime = cms.bool(True),
    firstValid = cms.vuint32(1)
)

StableParameters = cms.ESProducer("StableParametersTrivialProducer",

    # trigger decision
    
    # number of physics trigger algorithms
    NumberPhysTriggers = cms.uint32(512),

    # additional number of physics trigger algorithms
    NumberPhysTriggersExtended = cms.uint32(64),

    # number of technical triggers
    NumberTechnicalTriggers = cms.uint32(64),

    # trigger objects

    # muons
    NumberL1Mu = cms.uint32(4),
    
    # e/gamma and isolated e/gamma objects
    NumberL1IsoEG = cms.uint32(4),
    NumberL1NoIsoEG = cms.uint32(4),

    # central, forward and tau jets
    NumberL1CenJet = cms.uint32(4),
    NumberL1ForJet = cms.uint32(4),
    NumberL1TauJet = cms.uint32(4),

    # jet counts
    NumberL1JetCounts = cms.uint32(12),

    # hardware

    # number of maximum chips defined in the xml file
    NumberConditionChips = cms.uint32(1),

    # number of pins on the GTL condition chips
    PinsOnConditionChip = cms.uint32(512),

    # correspondence "condition chip - GTL algorithm word" in the hardware
    # e.g.: chip 2: 0 - 95;  chip 1: 96 - 128 (191)
    OrderConditionChip = cms.vint32(1),

    # number of PSB boards in GT
    NumberPsbBoards = cms.int32(7),

    # number of bits for eta of calorimeter objects
    IfCaloEtaNumberBits = cms.uint32(4),
    
    # number of bits for eta of muon objects
    IfMuEtaNumberBits = cms.uint32(6),
    
    # GT DAQ record organized in words of WordLength bits
    WordLength = cms.int32(64),

    # one unit in the word is UnitLength bits
    UnitLength = cms.int32(8)
)


