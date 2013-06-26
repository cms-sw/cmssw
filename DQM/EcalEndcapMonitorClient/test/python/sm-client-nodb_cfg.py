import FWCore.ParameterSet.Config as cms

process = cms.Process("ECALDQM")

process.load("DQM.EcalEndcapMonitorClient.EcalEndcapMonitorClient_cfi")

process.load("DQMServices.Core.DQM_cfg")

process.source = cms.Source("DQMHttpSource",
    sourceURL = cms.untracked.string('http://localhost.localdomain:40003/urn:xdaq-application:lid=44'),
    DQMconsumerName = cms.untracked.string('ECAL ENDCAP DQM Client'),
    DQMconsumerPriority = cms.untracked.string('normal'),
    topLevelFolderName = cms.untracked.string('EcalEndcap'),
    headerRetryInterval = cms.untracked.int32(5),
    maxDQMEventRequestRate = cms.untracked.double(1.0)
)

process.MessageLogger = cms.Service("MessageLogger",
    cout = cms.untracked.PSet(
        threshold = cms.untracked.string('WARNING'),
        default = cms.untracked.PSet(
            limit = cms.untracked.int32(0)
        ),
        noTimeStamps = cms.untracked.bool(True),
        noLineBreaks = cms.untracked.bool(True),
        EcalEndcapClient = cms.untracked.PSet(
            limit = cms.untracked.int32(0)
        )
    ),
    categories = cms.untracked.vstring('EcalEndcapClient'),
    destinations = cms.untracked.vstring('cout')
)

process.p = cms.Path(process.ecalEndcapMonitorClient)

process.ecalEndcapMonitorClient.location = 'H4'

process.DQM.collectorHost = ''

