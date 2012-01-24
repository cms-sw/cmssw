import FWCore.ParameterSet.Config as cms

hltBeamModeFilter = cms.EDFilter("HLTBeamModeFilter",
    #
    # InputTag for the L1 Global Trigger EVM readout record
    #   gtDigis        GT Emulator
    #   l1GtEvmUnpack  GT EVM Unpacker (default module name) 
    #   gtEvmDigis     GT EVM Unpacker in RawToDigi standard sequence  
    #
    #   cloned GT unpacker in HLT = gtEvmDigis
    L1GtEvmReadoutRecordTag = cms.InputTag("gtEvmDigis"),
    #
    # vector of allowed beam modes 
    # default value: 11 (STABLE)
    AllowedBeamMode = cms.vuint32(11),

    saveTags = cms.bool( False )
)
