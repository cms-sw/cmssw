import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")

process.load("FWCore.MessageService.MessageLogger_cfi")

process.MessageLogger.infos.placeholder = cms.untracked.bool(False)
process.MessageLogger.infos.threshold = cms.untracked.string("INFO")
process.MessageLogger.infos.default = cms.untracked.PSet(
    limit = cms.untracked.int32(1000000)
    )
process.MessageLogger.infos.FwkReport = cms.untracked.PSet(
#    reportEvery = cms.untracked.int32(10000),
    limit = cms.untracked.int32(10000000)
)

process.MessageLogger.cerr.threshold = cms.untracked.string("WARNING")



process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(2) )

process.source = cms.Source("EmptySource",
                            firstRun = cms.untracked.uint32(70170),
                            numberEventsInRun = cms.untracked.uint32(5)
                            )

process.load("DPGAnalysis.SiStripTools.apvlatency.fakeapvlatencyessource_cfi")

process.eessapvlatency = cms.ESSource("EmptyESSource",
                                     recordName = cms.string("APVLatencyRcd"),
                                     firstValid = cms.vuint32(1),
                                     iovIsRunNotTime = cms.bool(True)
                                     )

process.dummy = cms.EDAnalyzer("EventSetupRecordDataGetter",
                               verbose = cms.untracked.bool(True),
                               toGet = cms.VPSet(
    cms.PSet( record = cms.string("APVLatencyRcd"), data = cms.vstring("APVLatency"))
    )
                               )
  
process.p = cms.Path(process.dummy)
