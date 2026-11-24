import FWCore.ParameterSet.Config as cms

import Configuration.Geometry.defaultPhase2ConditionsEra_cff as _settings
import Geometry.MTDCommonData.defaultMTDConditionsEra_cff as _mtdgeo
_mtdgeo.check_mtdgeo()
_PH2_GLOBAL_TAG, _PH2_ERA = _settings.get_era_and_conditions(_mtdgeo.MTD_DEFAULT_VERSION)
from Configuration.ProcessModifiers.dd4hep_cff import dd4hep

process = cms.Process("BTLUNPACKERVALID",_PH2_ERA, dd4hep)

process.load('Configuration.StandardSequences.Services_cff')
process.load('Configuration.EventContent.EventContent_cff')
process.load('Configuration.StandardSequences.MagneticField_cff')
process.load('Geometry.MTDCommonData.GeometryDD4hepExtendedRun4MTDDefaultReco_cff')

process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(-1))

process.source = cms.Source("PoolSource",
                            fileNames = cms.untracked.vstring("file:step2.root")
                            )

process.load("Configuration.StandardSequences.Accelerators_cff")

# Step 1: packer (btlDigiToRaw), with the readout map
process.load('EventFilter.MTDRawToDigi.btlDigiToRaw_cff')

# Step 2: unpacker (btlRawToDigi), with the readout map and the electronics -> DetId mapping
process.load('EventFilter.MTDRawToDigi.btlRawToDigi_cff')

# Step 3: validator
process.btlValidator = cms.EDAnalyzer('BTLRawToDigiValidator',
    refDigis  = cms.InputTag("mix", "MTDBarrel"),   # original SortedCollection
    testDigis = cms.InputTag("btlRawToDigi"),       # BTLDigiSoA from the unpacker (digi -> raw -> digi)
)

process.path = cms.Path(
    process.btlDigiToRaw
    + process.btlRawToDigi
    + process.btlValidator
)

process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.MessageLogger.cerr.threshold = cms.untracked.string('INFO')
process.MessageLogger.cerr.INFO = cms.untracked.PSet(
    limit = cms.untracked.int32(0)
)
process.MessageLogger.cerr.BTLRawToDigiValidator = cms.untracked.PSet(
    #limit    = cms.untracked.int32(0),
    limit    = cms.untracked.int32(-1),
)

process.MessageLogger.cerr.threshold = "DEBUG"
process.MessageLogger.cerr.DEBUG = cms.untracked.PSet(
    limit = cms.untracked.int32(-1)
)

process.MessageLogger.debugModules = cms.untracked.vstring("btlRawToDigi")

process.out = cms.OutputModule("PoolOutputModule",
    outputCommands = cms.untracked.vstring('keep *'),
    fileName = cms.untracked.string('OutputBTLRawToDigiValidator.root')
)

process.options = cms.untracked.PSet(
    wantSummary = cms.untracked.bool(True)
    #accelerators = cms.untracked.vstring('cpu')
)

process.output = cms.EndPath(process.out)

