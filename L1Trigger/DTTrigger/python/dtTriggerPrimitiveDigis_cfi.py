import FWCore.ParameterSet.Config as cms

from L1TriggerConfig.DTTPGConfigProducers.L1DTTPGConfigFromDB_cff import *

dtTriggerPrimitiveDigis = cms.EDProducer("DTTrigProd",
    debug = cms.untracked.bool(False),
    # DT digis input tag
    digiTag = cms.InputTag("muonDTDigis"),
    # Convert output into DTTF sector numbering: 
    # false means [1-12] (useful for debug)
    # true is [0-11] useful as input for the DTTF emulator
    DTTFSectorNumbering = cms.bool(True),
    # config params for dumping of LUTs info from emulator
    lutBtic = cms.untracked.int32(31),
    lutDumpFlag = cms.untracked.bool(False)
)
