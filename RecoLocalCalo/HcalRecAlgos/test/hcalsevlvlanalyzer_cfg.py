import FWCore.ParameterSet.Config as cms

process = cms.Process("Demo")

process.load("FWCore.MessageService.MessageLogger_cfi")
process.MessageLogger = cms.Service("MessageLogger",
        destinations = cms.untracked.vstring("cout"),
        cout = cms.untracked.PSet(threshold = cms.untracked.string("INFO")))

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(1) )

#process.essource = cms.ESSource("EmptyESSource",
#   recordName = cms.string("HcalSeverityLevelComputerRcd"),
#   firstValid = cms.vuint32(1),
#   iovIsRunNotTime = cms.bool(True)
#)
#
#process.HcalRecAlgoESProducer = cms.ESProducer("HcalRecAlgoESProducer",
#    SeverityLevels = cms.VPSet(
#        cms.PSet( Level = cms.int32(0),
#                  RecHitFlags = cms.vstring(''),
#                  ChannelStatus = cms.vstring('')
#                ),
#        cms.PSet( Level = cms.int32(10),
#                  RecHitFlags = cms.vstring(''),
#                  ChannelStatus = cms.vstring('HcalCellHot')
#                ),
#        cms.PSet( Level = cms.int32(20),
#                  RecHitFlags = cms.vstring('HBHEHpdHitMultiplicity', 'HBHEPulseShape', 'HOBit',
#                                            'HFDigiTime', 'HFLongShort', 'ZDCBit', 'CalibrationBit'),
#                  ChannelStatus = cms.vstring('HcalCellOff', 'HcalCellDead')
#                )
#        )
#)
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
