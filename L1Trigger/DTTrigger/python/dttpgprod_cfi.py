import FWCore.ParameterSet.Config as cms

dttpgprod = cms.EDProducer("DTTrigProd",
    # Includes configuration parametersets
    #include "L1Trigger/DTTrigger/data/dttpg_conf.cff"
    # Switch off/on debug printouts for producer class
    debug = cms.untracked.bool(False),
    # Convert output into DTTF sector numbering: 
    # false means [1-12] (useful for debug)
    # true is [0-11] useful as input for the DTTF emulator
    DTTFSectorNumbering = cms.bool(True),
    # BX Correction used to set
    # correct BX to 0 in DTTF input
    BXOffset = cms.int32(16)
)


