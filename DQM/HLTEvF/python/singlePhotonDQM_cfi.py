import FWCore.ParameterSet.Config as cms

singlePhotonDQM = cms.EDFilter("HLTMonElectron",
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
        HLTCollectionLabels = cms.InputTag("hltL1IsoSinglePhotonL1MatchFilter"),
        IsoCollections = cms.VInputTag(cms.InputTag("none")),
        theHLTOutputTypes = cms.uint32(100)
    ), cms.PSet(
        PlotBounds = cms.vdouble(0.0, 0.0),
        HLTCollectionLabels = cms.InputTag("hltL1IsoSinglePhotonEtFilter"),
        IsoCollections = cms.VInputTag(cms.InputTag("none")),
        theHLTOutputTypes = cms.uint32(100)
    ), cms.PSet(
        PlotBounds = cms.vdouble(0.0, 10.0),
        HLTCollectionLabels = cms.InputTag("hltL1IsoSinglePhotonEcalIsolFilter"),
        IsoCollections = cms.VInputTag(cms.InputTag("l1IsolatedPhotonEcalIsol")),
        theHLTOutputTypes = cms.uint32(100)
    ), cms.PSet(
        PlotBounds = cms.vdouble(0.0, 10.0),
        HLTCollectionLabels = cms.InputTag("hltL1IsoSinglePhotonHcalIsolFilter"),
        IsoCollections = cms.VInputTag(cms.InputTag("l1IsolatedPhotonHcalIsol")),
        theHLTOutputTypes = cms.uint32(100)
    ), cms.PSet(
        PlotBounds = cms.vdouble(0.0, 10.0),
        HLTCollectionLabels = cms.InputTag("hltL1IsoSinglePhotonTrackIsolFilter"),
        IsoCollections = cms.VInputTag(cms.InputTag("l1IsoPhotonTrackIsol")),
        theHLTOutputTypes = cms.uint32(91)
    )),
    disableROOToutput = cms.untracked.bool(True),
    PtMax = cms.untracked.double(200.0)
)


