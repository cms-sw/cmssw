import FWCore.ParameterSet.Config as cms
import sys

process = cms.Process("RawToDigi")

process.load("FWCore.MessageService.MessageLogger_cfi")

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(100) )

# Input source
process.source = cms.Source("NewEventStreamFileReader",
    fileNames = cms.untracked.vstring( 'file:'+sys.argv[-1])
)

#process.load('CalibTracker.SiStripESProducers.fake.Phase2TrackerConfigurableCablingESSource_cfi')
process.load('TestbeamCabling_cfi')
process.load('EventFilter.Phase2TrackerRawToDigi.Phase2TrackerDigiProducer_cfi')
# process.load('EventFilter.Phase2TrackerRawToDigi.Phase2TrackerCommissioningDigiProducer_cfi')

process.out = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string('myOutputFile.root')
)

# process.output = cms.OutputModule(
#     "PoolOutputModule",
#     fileName = cms.untracked.string('output.root'),
#     outputCommands = cms.untracked.vstring(
#       'drop *',
#       'keep *_*_*_DigiToRawToDigi'
#       )
#     )
  
process.p = cms.Path(process.Phase2TrackerDigitestproducer)
# process.p = cms.Path(process.Phase2TrackerDigitestproducer*process.Phase2TrackerDigiCondDataproducer)

process.e = cms.EndPath(process.out)
