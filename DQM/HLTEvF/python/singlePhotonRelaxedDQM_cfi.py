import FWCore.ParameterSet.Config as cms

singlePhotonRelaxedDQM = cms.EDFilter("HLTMonElectron",
    outputFile = cms.untracked.string('./L1TDQM.root'),
    MonitorDaemon = cms.untracked.bool(True),
    reqNum = cms.uint32(1),
    DaqMonitorBEInterface = cms.untracked.bool(True),
    filters = cms.VPSet(cms.PSet(
        PlotBounds = cms.vdouble(0.0, 0.0),
        HLTCollectionLabels = cms.InputTag("l1seedRelaxedSingle"),
        IsoCollections = cms.VInputTag(cms.InputTag("none")),
        theHLTOutputTypes = cms.uint32(82)
    ), cms.PSet(
        PlotBounds = cms.vdouble(0.0, 0.0),
        HLTCollectionLabels = cms.InputTag("hltL1NonIsoSinglePhotonL1MatchFilterRegional"),
        IsoCollections = cms.VInputTag(cms.InputTag("none")),
        theHLTOutputTypes = cms.uint32(100)
    ), cms.PSet(
        PlotBounds = cms.vdouble(0.0, 0.0),
        HLTCollectionLabels = cms.InputTag("hltL1NonIsoSinglePhotonEtFilter"),
        IsoCollections = cms.VInputTag(cms.InputTag("none")),
        theHLTOutputTypes = cms.uint32(100)
    ), cms.PSet(
        PlotBounds = cms.vdouble(0.0, 10.0),
        HLTCollectionLabels = cms.InputTag("hltL1NonIsoSinglePhotonEcalIsolFilter"),
        IsoCollections = cms.VInputTag(cms.InputTag("l1IsolatedPhotonEcalIsol"), cms.InputTag("l1NonIsolatedPhotonEcalIsol")),
        theHLTOutputTypes = cms.uint32(100)
    ), cms.PSet(
        PlotBounds = cms.vdouble(0.0, 10.0),
        HLTCollectionLabels = cms.InputTag("hltL1NonIsoSinglePhotonHcalIsolFilter"),
        IsoCollections = cms.VInputTag(cms.InputTag("l1IsolatedPhotonHcalIsol"), cms.InputTag("l1NonIsolatedPhotonHcalIsol")),
        theHLTOutputTypes = cms.uint32(100)
    ), cms.PSet(
        PlotBounds = cms.vdouble(0.0, 10.0),
        HLTCollectionLabels = cms.InputTag("hltL1NonIsoSinglePhotonTrackIsolFilter"),
        IsoCollections = cms.VInputTag(cms.InputTag("l1IsoPhotonTrackIsol"), cms.InputTag("l1NonIsoPhotonTrackIsol")),
        theHLTOutputTypes = cms.uint32(91)
    )),
    disableROOToutput = cms.untracked.bool(True),
    PtMax = cms.untracked.double(200.0)
)


