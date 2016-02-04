import FWCore.ParameterSet.Config as cms

process = cms.Process("analyzer")

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(100) )

process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.MessageLogger.cout.placeholder = cms.untracked.bool(False)
process.MessageLogger.cout.threshold = cms.untracked.string('INFO')
process.MessageLogger.debugModules = cms.untracked.vstring('*')

process.source = cms.Source("DaqSource",
    readerPluginName = cms.untracked.string('CSCFileReader'),
    readerPset = cms.untracked.PSet(
        firstEvent  = cms.untracked.int32(0),
        tfDDUnumber = cms.untracked.int32(0),
        FED760 = cms.untracked.vstring('RUI01'),
        RUI01  = cms.untracked.vstring('/tmp/kkotov/66637_.bin_760')
  )
)

#process.source = cms.Source("PoolSource",
#  fileNames = cms.untracked.vstring('')
#)
#readFiles = cms.untracked.vstring()
#process.source = cms.Source ("PoolSource", fileNames = readFiles)
#readFiles.extend((
#        '/castor/cern.ch/cms/store/data/Commissioning08/Cosmics/RAW/v1/000/066/615/',
#))

# CSC TF (copy-paste L1Trigger/Configuration/python/L1RawToDigi_cff.py + little modifications)
import EventFilter.CSCTFRawToDigi.csctfunpacker_cfi
process.csctfDigis = EventFilter.CSCTFRawToDigi.csctfunpacker_cfi.csctfunpacker.clone()
process.csctfDigis.producer = 'source'
#

# CSC Track Finder emulator (copy-paste from L1Trigger/Configuration/python/SimL1Emulator_cff.py + little modifications)
import L1Trigger.CSCTrackFinder.csctfTrackDigis_cfi
process.simCsctfTrackDigis = L1Trigger.CSCTrackFinder.csctfTrackDigis_cfi.csctfTrackDigis.clone()
import L1Trigger.CSCTrackFinder.csctfDigis_cfi
process.simCsctfTrackDigis.SectorReceiverInput = cms.untracked.InputTag("csctfDigis")
process.simCsctfTrackDigis.useDT = cms.bool(False)
process.simCsctfTrackDigis.initializeFromPSet = cms.bool(True)
# Following important parameters have to be set for singles
process.simCsctfTrackDigis.SectorProcessor.trigger_on_ME1a = cms.bool(True)
process.simCsctfTrackDigis.SectorProcessor.trigger_on_ME1b = cms.bool(True)
process.simCsctfTrackDigis.SectorProcessor.trigger_on_ME2  = cms.bool(True)
process.simCsctfTrackDigis.SectorProcessor.trigger_on_ME3  = cms.bool(True)
process.simCsctfTrackDigis.SectorProcessor.trigger_on_ME4  = cms.bool(True)
process.simCsctfTrackDigis.SectorProcessor.trigger_on_MB1a = cms.bool(False)
process.simCsctfTrackDigis.SectorProcessor.trigger_on_MB1d = cms.bool(False)
process.simCsctfTrackDigis.SectorProcessor.singlesTrackPt     = cms.uint32(255)
process.simCsctfTrackDigis.SectorProcessor.singlesTrackOutput = cms.uint32(1)
#

# Little pieces of configuration, taken here and there
process.load("Alignment.CommonAlignmentProducer.GlobalPosition_Fake_cff")
process.load("Geometry.MuonNumbering.muonNumberingInitialization_cfi")
process.load("Geometry.MuonCommonData.muonIdealGeometryXML_cfi")
process.load("Geometry.CSCGeometry.cscGeometry_cfi")
process.load("L1TriggerConfig.CSCTFConfigProducers.L1CSCTFConfig_cff")
process.load("L1TriggerConfig.L1ScalesProducers.L1MuTriggerScalesConfig_cff")
process.load("L1TriggerConfig.L1ScalesProducers.L1MuTriggerPtScaleConfig_cff")

process.csctfanalyzer = cms.EDAnalyzer("CSCTFAnalyzer",
         mbProducer     = cms.untracked.InputTag("null"),
         lctProducer    = cms.untracked.InputTag("csctfDigis"),
#         trackProducer  = cms.untracked.InputTag("csctfDigis"),
         trackProducer  = cms.untracked.InputTag("simCsctfTrackDigis"),
         statusProducer = cms.untracked.InputTag("null")
)

#process.out = cms.OutputModule("PoolOutputModule",
#  fileName = cms.untracked.string("qwe.root"),
#)

process.p = cms.Path(process.csctfDigis*process.simCsctfTrackDigis*process.csctfanalyzer)
#process.outpath = cms.EndPath(process.out)
#process.schedule = cms.Schedule(process.p,process.outpath)

