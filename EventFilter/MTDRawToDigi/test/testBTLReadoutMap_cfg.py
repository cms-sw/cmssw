import FWCore.ParameterSet.Config as cms

import Configuration.Geometry.defaultPhase2ConditionsEra_cff as _settings
import Geometry.MTDCommonData.defaultMTDConditionsEra_cff as _mtdgeo
_mtdgeo.check_mtdgeo()
_PH2_GLOBAL_TAG, _PH2_ERA = _settings.get_era_and_conditions(_mtdgeo.MTD_DEFAULT_VERSION)
from Configuration.ProcessModifiers.dd4hep_cff import dd4hep

process = cms.Process("TEST",_PH2_ERA, dd4hep)

process.source = cms.Source("EmptySource")

process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(1))

process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.MessageLogger.cerr.threshold = cms.untracked.string('INFO')
process.MessageLogger.cerr.INFO = cms.untracked.PSet(
    #limit = cms.untracked.int32(0)
    limit = cms.untracked.int32(-1)
)
process.MessageLogger.cerr.TestBTLElectronicsMapping = cms.untracked.PSet(
    #limit = cms.untracked.int32(0)
    limit = cms.untracked.int32(-1)
)


process.load("Geometry.MTDNumberingBuilder.mtdTopology_cfi")
process.load('Geometry.MTDCommonData.GeometryDD4hepExtendedRun4MTDDefaultReco_cff')

# ESProducer
process.btlReadoutMapESProducer = cms.ESProducer(
    "BTLReadoutMapESProducer"
)

# your test analyzer
process.test = cms.EDAnalyzer("TestBTLReadoutMap",
                              DDDetector = cms.ESInputTag('',''),
                              ddTopNodeName = cms.untracked.string('BarrelTimingLayer')
)

process.p = cms.Path(process.test)
