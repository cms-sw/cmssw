import FWCore.ParameterSet.Config as cms

dtTriggerPrimitiveDigis = cms.EDProducer("DTTrigProd",
    debug = cms.untracked.bool(False),
    tTrigModeConfig = cms.PSet(
        vPropWire = cms.double(24.4),
        doTOFCorrection = cms.bool(False),
        tofCorrType = cms.int32(1),
        kFactor = cms.double(-2.0), ##retuned in CMSSW15X

        wirePropCorrType = cms.int32(1),
        doWirePropCorrection = cms.bool(False),
        doT0Correction = cms.bool(True), ##FIXME: remove, not anymore needed from CMSSW180pre1

        debug = cms.untracked.bool(False)
    ),
    # Synchronizer related stuff
    tTrigMode = cms.string('DTTTrigSyncFromDB'),
    # Convert output into DTTF sector numbering: 
    # false means [1-12] (useful for debug)
    # true is [0-11] useful as input for the DTTF emulator
    DTTFSectorNumbering = cms.bool(True)
)


