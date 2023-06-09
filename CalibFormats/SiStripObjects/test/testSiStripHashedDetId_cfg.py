import FWCore.ParameterSet.Config as cms
from Configuration.Eras.Era_Run3_cff import Run3
process = cms.Process("testSiStripHashedDetId", Run3)

process.load("Configuration.Geometry.GeometryRecoDB_cff")

process.load('FWCore.MessageService.MessageLogger_cfi')   
process.MessageLogger.cerr.enable = True
process.MessageLogger.cerr.threshold = cms.untracked.string('DEBUG')
process.MessageLogger.SiStripDqmCommon=dict()  
process.MessageLogger.cout = cms.untracked.PSet(
    enable = cms.untracked.bool(True),
    threshold = cms.untracked.string("INFO"),
    default   = cms.untracked.PSet(limit = cms.untracked.int32(0)),                       
    FwkReport = cms.untracked.PSet(limit = cms.untracked.int32(-1),
                                   reportEvery = cms.untracked.int32(1000)
                                   ),                                                      
    SiStripDqmCommon = cms.untracked.PSet( limit = cms.untracked.int32(-1)),
    enableStatistics = cms.untracked.bool(True)
    )

process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:phase1_2022_realistic', '')

process.source = cms.Source("EmptySource",
                            firstRun = cms.untracked.uint32(1),
                            numberEventsInRun = cms.untracked.uint32(1),
                            numberEventsInLuminosityBlock = cms.untracked.uint32(1))

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)

process.test =  cms.EDAnalyzer("testSiStripHashedDetId")

process.p = cms.Path(process.test)
