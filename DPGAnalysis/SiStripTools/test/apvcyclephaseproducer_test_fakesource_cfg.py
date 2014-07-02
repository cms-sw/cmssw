import FWCore.ParameterSet.Config as cms

process = cms.Process("APVCyclePhaseProducerTestFakeSource")

process.load("FWCore.MessageService.MessageLogger_cfi")

process.MessageLogger.infos.placeholder = cms.untracked.bool(False)
process.MessageLogger.infos.threshold = cms.untracked.string("INFO")
process.MessageLogger.infos.default = cms.untracked.PSet(
    limit = cms.untracked.int32(10000000)
    )
process.MessageLogger.infos.FwkReport = cms.untracked.PSet(
    reportEvery = cms.untracked.int32(10000)
    )
process.MessageLogger.cerr.threshold = cms.untracked.string("WARNING")

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(10) )

process.source = cms.Source("PoolSource",
                    fileNames = cms.untracked.vstring("/store/data/Commissioning2013/Cosmics/RAW/v1/00000/007C8D94-9A46-E311-9596-003048F0E594.root",
                                                      "/store/data/Commissioning2013/Cosmics/RAW/v1/00000/06631F08-9A45-E311-9BFD-003048F17496.root"
),
#                    skipBadFiles = cms.untracked.bool(True),
                    inputCommands = cms.untracked.vstring("keep *", "drop *_MEtoEDMConverter_*_*")
                    )


#process.source = cms.Source("EmptySource",
#                            firstRun = cms.untracked.uint32(216322),
#                            numberEventsInRun = cms.untracked.uint32(10)
#                            )

process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.GlobalTag.globaltag = "GR_R_70_V1::All"
#-------------------------------------------------------------------------
process.load("DPGAnalysis.SiStripTools.SiStripConfObjectAPVPhaseOffsetsFakeESSource_cfi")
#-------------------------------------------------------------------------

process.load("Configuration.StandardSequences.RawToDigi_Data_cff")
process.load("Configuration.StandardSequences.MagneticField_AutoFromDBCurrent_cff")
process.load("Configuration.StandardSequences.GeometryDB_cff")
process.load("Configuration.StandardSequences.Reconstruction_cff")

#process.load("DPGAnalysis.SiStripTools.configurableapvcyclephaseproducer_GR09_withdefault_cff")
process.load("DPGAnalysis.SiStripTools.apvcyclephaseproducerfroml1tsDB_cfi")
#process.APVPhases.recordLabel = cms.untracked.string("")

process.load("DPGAnalysis.SiStripTools.apvcyclephasemonitor_cfi")

process.TFileService = cms.Service('TFileService',
                                   fileName = cms.string('apvcyclephaseproducer_test_fakesource.root')
                                   )

process.p0 = cms.Path(process.scalersRawToDigi + process.APVPhases +
                      process.apvcyclephasemonitor )

