import FWCore.ParameterSet.Config as cms

#from Configuration.Eras.Era_Run2_2018_cff import Run2_2018
from Configuration.Eras.Era_Run3_cff import Run3

#process = cms.Process('HcalGeometryTest',Run2_2018)
process = cms.Process('HcalGeometryTest',Run3)

# import of standard configurations
process.load('Configuration.StandardSequences.Services_cff')
process.load('FWCore.MessageService.MessageLogger_cfi')
#process.load('Configuration.Geometry.GeometryExtended2018Reco_cff')
process.load('Configuration.Geometry.GeometryExtended2021Reco_cff')
process.load('Geometry.HcalTowerAlgo.hcalGeometryDetIdTester_cfi')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
from Configuration.AlCa.GlobalTag import GlobalTag
#process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:phase1_2018_realistic', '')
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:phase1_2022_realistic', '')

process.source = cms.Source("EmptySource")
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
    )

process.hcalGeometryDetIdTester.DetectorMin = 1
#process.hcalGeometryDetIdTester.DetectorMin = 2
process.hcalGeometryDetIdTester.DetectorMax = 2

process.Timing = cms.Service("Timing")
process.SimpleMemoryCheck = cms.Service("SimpleMemoryCheck")

process.p1 = cms.Path(process.hcalGeometryDetIdTester)
