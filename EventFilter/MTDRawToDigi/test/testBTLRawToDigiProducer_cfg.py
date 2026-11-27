import FWCore.ParameterSet.Config as cms

import Configuration.Geometry.defaultPhase2ConditionsEra_cff as _settings
import Geometry.MTDCommonData.defaultMTDConditionsEra_cff as _mtdgeo
_mtdgeo.check_mtdgeo()
_PH2_GLOBAL_TAG, _PH2_ERA = _settings.get_era_and_conditions(_mtdgeo.MTD_DEFAULT_VERSION)
from Configuration.ProcessModifiers.dd4hep_cff import dd4hep

process = cms.Process("BTLUNPACKER",_PH2_ERA, dd4hep)

process.load('Configuration.StandardSequences.Services_cff')
process.load('Configuration.EventContent.EventContent_cff')
process.load('Configuration.StandardSequences.MagneticField_cff')

process.load('Geometry.MTDCommonData.GeometryDD4hepExtendedRun4MTDDefaultReco_cff')

process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(-1))

process.source = cms.Source("PoolSource",
                            fileNames = cms.untracked.vstring("file:BtlDigiToRaw.root")
)

process.load("Configuration.StandardSequences.Accelerators_cff")

# unpacker, with the readout map and the electronics -> DetId mapping
process.load('EventFilter.MTDRawToDigi.btlRawToDigi_cff')


process.load("FWCore.MessageLogger.MessageLogger_cfi")
#process.MessageLogger.cerr.threshold = "INFO"
process.MessageLogger.cerr.threshold = "DEBUG"
process.MessageLogger.cerr.DEBUG = cms.untracked.PSet(
    limit = cms.untracked.int32(-1)
)

process.MessageLogger.debugModules = cms.untracked.vstring("btlRawToDigi")

process.out = cms.OutputModule("PoolOutputModule",
                               splitLevel = cms.untracked.int32(0),
                               outputCommands = cms.untracked.vstring('keep *'),
                               fileName = cms.untracked.string('BtlRawToDigi.root')
)

process.options = cms.untracked.PSet(
    wantSummary = cms.untracked.bool(True)
    #accelerators = cms.untracked.vstring('cpu')
)

process.path1 = cms.Path(process.btlRawToDigi)
process.output = cms.EndPath(process.out)
