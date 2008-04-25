import FWCore.ParameterSet.Config as cms

dtTriggerPrimitiveDigis = cms.EDProducer("DTTrigProd",
    debug = cms.untracked.bool(False),
    tTrigModeConfig = cms.PSet(
        debug = cms.untracked.bool(False),
        kFactor = cms.double(-2.0), ##retuned in CMSSW15X

        vPropWire = cms.double(24.4),
        tofCorrType = cms.int32(1),
        tTrig = cms.double(500.0)
    ),
    # DT digis input tag
    digiTag = cms.InputTag("muonDTDigis"),
    # Synchronizer related stuff
    tTrigMode = cms.string('DTTTrigSyncTOFCorr'),
    # Convert output into DTTF sector numbering: 
    # false means [1-12] (useful for debug)
    # true is [0-11] useful as input for the DTTF emulator
    DTTFSectorNumbering = cms.bool(True)
)



