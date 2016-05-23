#
# WARNING: This file is in the L1T configuration critical path.
#
# All changes must be explicitly discussed with the L1T offline coordinator.
#
import FWCore.ParameterSet.Config as cms

GlobalParametersRcdSource = cms.ESSource("EmptyESSource",
    recordName = cms.string('L1TGlobalParametersRcd'),
    iovIsRunNotTime = cms.bool(True),
    firstValid = cms.vuint32(1)
)

#GlobalParameters = cms.ESProducer("L1TGlobalParamsESProducer", 
GlobalParameters = cms.ESProducer("StableParametersTrivialProducer", 
    # bx in event
    #NumberBxInEvent = cms.int32(5),

    # trigger decision
    
    # number of physics trigger algorithms
    NumberPhysTriggers = cms.uint32(512),


    # trigger objects

    # muons
    NumberL1Muon = cms.uint32(12),
    
    # e/gamma and isolated e/gamma objects
    NumberL1EGamma = cms.uint32(12),

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
)


