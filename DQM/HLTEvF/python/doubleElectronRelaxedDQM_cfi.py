import FWCore.ParameterSet.Config as cms

doubleElectronRelaxedDQM = cms.EDFilter("HLTMonElectron",
    outputFile = cms.untracked.string('./L1TDQM.root'),
    MonitorDaemon = cms.untracked.bool(True),
    reqNum = cms.uint32(2),
    DaqMonitorBEInterface = cms.untracked.bool(True),
    filters = cms.VPSet(cms.PSet(
        PlotBounds = cms.vdouble(0.0, 0.0),
        HLTCollectionLabels = cms.InputTag("l1seedRelaxedDouble"),
        IsoCollections = cms.VInputTag(cms.InputTag("none")),
        theHLTOutputTypes = cms.uint32(82)
    ), cms.PSet(
        PlotBounds = cms.vdouble(0.0, 0.0),
        HLTCollectionLabels = cms.InputTag("hltL1NonIsoDoubleElectronL1MatchFilterRegional"),
        IsoCollections = cms.VInputTag(cms.InputTag("none")),
        theHLTOutputTypes = cms.uint32(100)
    ), cms.PSet(
        PlotBounds = cms.vdouble(0.0, 0.0),
        HLTCollectionLabels = cms.InputTag("hltL1NonIsoDoubleElectronEtFilter"),
        IsoCollections = cms.VInputTag(cms.InputTag("none")),
        theHLTOutputTypes = cms.uint32(100)
    ), cms.PSet(
        PlotBounds = cms.vdouble(0.0, 10.0),
        HLTCollectionLabels = cms.InputTag("hltL1NonIsoDoubleElectronHcalIsolFilter"),
        IsoCollections = cms.VInputTag(cms.InputTag("l1IsolatedElectronHcalIsol"), cms.InputTag("l1NonIsolatedElectronHcalIsol")),
        theHLTOutputTypes = cms.uint32(100)
    ), cms.PSet(
        PlotBounds = cms.vdouble(0.0, 0.0),
        HLTCollectionLabels = cms.InputTag("hltL1NonIsoDoubleElectronPixelMatchFilter"),
        IsoCollections = cms.VInputTag(cms.InputTag("none")),
        theHLTOutputTypes = cms.uint32(100)
    ), cms.PSet(
        PlotBounds = cms.vdouble(0.0, 0.0),
        HLTCollectionLabels = cms.InputTag("hltL1NonIsoDoubleElectronEoverpFilter"),
        IsoCollections = cms.VInputTag(cms.InputTag("none")),
        theHLTOutputTypes = cms.uint32(92)
    ), cms.PSet(
        PlotBounds = cms.vdouble(0.0, 10.0),
        HLTCollectionLabels = cms.InputTag("hltL1NonIsoDoubleElectronTrackIsolFilter"),
        IsoCollections = cms.VInputTag(cms.InputTag("l1IsoElectronTrackIsol"), cms.InputTag("l1NonIsoElectronTrackIsol")),
        theHLTOutputTypes = cms.uint32(92)
    )),
    disableROOToutput = cms.untracked.bool(True),
    PtMax = cms.untracked.double(200.0)
)


