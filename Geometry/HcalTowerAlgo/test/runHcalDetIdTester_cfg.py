import FWCore.ParameterSet.Config as cms
from Configuration.StandardSequences.Eras import eras

process = cms.Process('HcalGeometryTest',eras.Run3)

# import of standard configurations
process.load('Configuration.StandardSequences.Services_cff')
process.load('FWCore.MessageService.MessageLogger_cfi')
#process.load("Geometry.HcalCommonData.testGeometry17bXML_cfi")
#process.load("Geometry.HcalCommonData.hcalDDConstants_cff")
#process.load("Geometry.HcalEventSetup.hcalTopologyIdeal_cfi")
process.load('Configuration.StandardSequences.GeometryRecoDB_cff')
process.load('Configuration.StandardSequences.GeometrySimDB_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')

from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:phase1_2021_realistic', '')

process.source = cms.Source("EmptySource")
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)

process.hga = cms.EDAnalyzer("HcalDetIdTester",
                             GeometryFromDB = cms.bool(True)
#                            GeometryFromDB = cms.bool(False)
)

process.Timing = cms.Service("Timing")

process.p1 = cms.Path(process.hga)
