import FWCore.ParameterSet.Config as cms

electronDQMConsumer = cms.EDAnalyzer("HLTMonElectronConsumer",
    outputFile = cms.untracked.string('./L1TDQMConsumer.root'),
    MonitorDaemon = cms.untracked.bool(True),
    PixelTag = cms.InputTag("electronDQMPixelMatch"),
    DaqMonitorBEInterface = cms.untracked.bool(True),
    IsoTag = cms.InputTag("electronDQMIsoDist"),
    disableROOToutput = cms.untracked.bool(True)
)


