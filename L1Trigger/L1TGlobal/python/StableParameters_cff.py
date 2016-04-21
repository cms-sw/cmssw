#
# WARNING: This file is in the L1T configuration critical path.
#
# All changes must be explicitly discussed with the L1T offline coordinator.
#
import FWCore.ParameterSet.Config as cms

StableParametersRcdSource = cms.ESSource("EmptyESSource",
    recordName = cms.string('L1TGlobalParametersRcd'),
    iovIsRunNotTime = cms.bool(True),
    firstValid = cms.vuint32(1)
)

StableParameters = cms.ESProducer("L1TGlobalParamsESProducer",

 
    # bx in event
    NumberBxInEvent = cms.int32(5),

    # trigger decision
    
    # number of physics trigger algorithms
    NumberPhysTriggers = cms.uint32(512),


    # trigger objects

    # muons
    NumberL1Mu = cms.uint32(12),
    
    # e/gamma and isolated e/gamma objects
    NumberL1EG = cms.uint32(12),

    #  jets
    NumberL1Jet = cms.uint32(12),

    # taus
    NumberL1Tau = cms.uint32(8),

    # hardware

    # number of maximum chips defined in the xml file
    NumberChips = cms.uint32(1),

    # number of pins on the GTL condition chips
    PinsOnChip = cms.uint32(512),

    # correspondence "condition chip - GTL algorithm word" in the hardware
    # e.g.: chip 2: 0 - 95;  chip 1: 96 - 128 (191)
    OrderOfChip = cms.vint32(1),

    # number of PSB boards in GT
    #NumberPsbBoards = cms.int32(7),

    # number of bits for eta of calorimeter objects
    #IfCaloEtaNumberBits = cms.uint32(4),
    
    # number of bits for eta of muon objects
    #IfMuEtaNumberBits = cms.uint32(6),
    
    # GT DAQ record organized in words of WordLength bits
    #WordLength = cms.int32(64),

    # one unit in the word is UnitLength bits
    #UnitLength = cms.int32(8)
)


