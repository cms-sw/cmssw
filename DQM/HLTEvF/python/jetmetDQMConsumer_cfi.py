import FWCore.ParameterSet.Config as cms

jetmetDQMConsumer = cms.EDAnalyzer("HLTMonJetMETConsumer",
    outputFile = cms.untracked.string('./L1TDQMConsumer.root'),
    MonitorDaemon = cms.untracked.bool(True),
    PixelTag = cms.InputTag("electronDQMPixelMatch"),
    DaqMonitorBEInterface = cms.untracked.bool(True),
    IsoTag = cms.InputTag("electronDQMIsoDist"),
    disableROOToutput = cms.untracked.bool(True),
    reffilters = cms.VPSet(
      cms.PSet(HLTRefLabels = cms.string("hltL1s1Level1jet15")),
      cms.PSet(HLTRefLabels = cms.string("hltL1s1jet30")),
      cms.PSet(HLTRefLabels = cms.string("hlt1jet30"))
    ),
    probefilters = cms.VPSet(
      cms.PSet(HLTProbeLabels = cms.string("hltL1s1Level1jet15")),
      cms.PSet(HLTProbeLabels = cms.string("hltL1s1jet30")),
      cms.PSet(HLTProbeLabels = cms.string("hlt1jet30"))
    )
)
