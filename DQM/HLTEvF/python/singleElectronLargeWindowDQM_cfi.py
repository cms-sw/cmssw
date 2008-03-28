import FWCore.ParameterSet.Config as cms

singleElectronLargeWindowDQM = cms.EDFilter("HLTMonElectron",
    outputFile = cms.untracked.string('./L1TDQM.root'),
    MonitorDaemon = cms.untracked.bool(True),
    reqNum = cms.uint32(1),
    DaqMonitorBEInterface = cms.untracked.bool(True),
    filters = cms.VPSet(cms.PSet(
        PlotBounds = cms.vdouble(0.0, 0.0),
        HLTCollectionLabels = cms.InputTag("l1seedSingle"),
        IsoCollections = cms.VInputTag(cms.InputTag("none")),
        theHLTOutputTypes = cms.uint32(83)
    ), cms.PSet(
        PlotBounds = cms.vdouble(0.0, 0.0),
        HLTCollectionLabels = cms.InputTag("hltL1IsoLargeWindowSingleL1MatchFilter"),
        IsoCollections = cms.VInputTag(cms.InputTag("none")),
        theHLTOutputTypes = cms.uint32(100)
    ), cms.PSet(
        PlotBounds = cms.vdouble(0.0, 0.0),
        HLTCollectionLabels = cms.InputTag("hltL1IsoLargeWindowSingleElectronEtFilter"),
        IsoCollections = cms.VInputTag(cms.InputTag("none")),
        theHLTOutputTypes = cms.uint32(100)
    ), cms.PSet(
        PlotBounds = cms.vdouble(0.0, 10.0),
        HLTCollectionLabels = cms.InputTag("hltL1IsoLargeWindowSingleElectronHcalIsolFilter"),
        IsoCollections = cms.VInputTag(cms.InputTag("l1IsolatedElectronHcalIsol")),
        theHLTOutputTypes = cms.uint32(100)
    ), cms.PSet(
        PlotBounds = cms.vdouble(0.0, 0.0),
        HLTCollectionLabels = cms.InputTag("hltL1IsoLargeWindowSingleElectronPixelMatchFilter"),
        IsoCollections = cms.VInputTag(cms.InputTag("none")),
        theHLTOutputTypes = cms.uint32(100)
    ), cms.PSet(
        PlotBounds = cms.vdouble(0.0, 0.0),
        HLTCollectionLabels = cms.InputTag("hltL1IsoLargeWindowSingleElectronHOneOEMinusOneOPFilter"),
        IsoCollections = cms.VInputTag(cms.InputTag("none")),
        theHLTOutputTypes = cms.uint32(92)
    ), cms.PSet(
        PlotBounds = cms.vdouble(0.0, 10.0),
        HLTCollectionLabels = cms.InputTag("hltL1IsoLargeWindowSingleElectronTrackIsolFilter"),
        IsoCollections = cms.VInputTag(cms.InputTag("l1IsoLargeWindowElectronTrackIsol")),
        theHLTOutputTypes = cms.uint32(92)
    )),
    disableROOToutput = cms.untracked.bool(True),
    PtMax = cms.untracked.double(200.0)
)


