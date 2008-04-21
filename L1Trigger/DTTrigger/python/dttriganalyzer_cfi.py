import FWCore.ParameterSet.Config as cms

from L1Trigger.DTTrigger.dttpg_conf_cff import *
dttriganalyzer = cms.EDFilter("DTTrigTest",
    # Incude configuration parametersets
    DTTPGParametersBlock,
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
    # Output filename
    outputFileName = cms.untracked.string('DTTPG_test.root')
)


