import FWCore.ParameterSet.Config as cms

process = cms.Process("Demo")

process.load("FWCore.MessageService.MessageLogger_cfi")
process.MessageLogger = cms.Service("MessageLogger",
    cerr = cms.untracked.PSet(
        enable = cms.untracked.bool(False)
    ),
    cout = cms.untracked.PSet(
        enable = cms.untracked.bool(True),
        threshold = cms.untracked.string('INFO')
    )
)

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(1) )


process.load("RecoLocalCalo.HcalRecAlgos.hcalRecAlgoESProd_cfi")



process.source = cms.Source("EmptySource",
    numberEventsInRun = cms.untracked.uint32(1),
    firstRun = cms.untracked.uint32(1)
)

process.demo = cms.EDAnalyzer("HcalSevLvlAnalyzer")

## the following is just a test module
#process.get = cms.EDAnalyzer("EventSetupRecordDataGetter",
#     toGet = cms.VPSet(cms.PSet(
#         record = cms.string('HcalSeverityLevelComputerRcd'),
#         data = cms.vstring('HcalSeverityLevelComputer')
#     )),
#     verbose = cms.untracked.bool(True)
#)
#process.p = cms.Path(process.get)


process.p = cms.Path(process.demo)
