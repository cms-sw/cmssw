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

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(40) )

process.source = cms.Source("EmptySource",
                            firstRun = cms.untracked.uint32(70414),
                            numberEventsInRun = cms.untracked.uint32(10)
                            )

#-------------------------------------------------------------------------

#process.load("DPGAnalysis.SiStripTools.configurableapvcyclephaseproducer_GR09_withdefault_cff")
process.load("DPGAnalysis.SiStripTools.configurableapvcyclephaseproducer_CRAFT08_cfi")

process.load("DPGAnalysis.SiStripTools.apvcyclephasemonitor_cfi")

process.TFileService = cms.Service('TFileService',
                                   fileName = cms.string('apvcyclephaseproducer_test_fakesource.root')
                                   )

process.p0 = cms.Path(process.APVPhases +
                      process.apvcyclephasemonitor )

