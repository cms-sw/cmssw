import FWCore.ParameterSet.Config as cms

doublePhotonRelaxedDQM = cms.EDFilter("HLTMonElectron",
    outputFile = cms.untracked.string('./L1TDQM.root'),
    MonitorDaemon = cms.untracked.bool(True),
    reqNum = cms.uint32(2),
    DaqMonitorBEInterface = cms.untracked.bool(True),
    filters = cms.VPSet(cms.PSet(
        PlotBounds = cms.vdouble(0.0, 0.0),
        HLTCollectionLabels = cms.InputTag("l1seedRelaxedDouble"),
        IsoCollections = cms.VInputTag(cms.InputTag("none")),
        theHLTOutputTypes = cms.uint32(82)
    ), 
        cms.PSet(
            PlotBounds = cms.vdouble(0.0, 0.0),
            HLTCollectionLabels = cms.InputTag("hltL1NonIsoDoublePhotonL1MatchFilterRegional"),
            IsoCollections = cms.VInputTag(cms.InputTag("none")),
            theHLTOutputTypes = cms.uint32(100)
        ), 
        cms.PSet(
            PlotBounds = cms.vdouble(0.0, 0.0),
            HLTCollectionLabels = cms.InputTag("hltL1NonIsoDoublePhotonEtFilter"),
            IsoCollections = cms.VInputTag(cms.InputTag("none")),
            theHLTOutputTypes = cms.uint32(100)
        ), 
        cms.PSet(
            PlotBounds = cms.vdouble(0.0, 10.0),
            HLTCollectionLabels = cms.InputTag("hltL1NonIsoDoublePhotonEcalIsolFilter"),
            IsoCollections = cms.VInputTag(cms.InputTag("l1IsolatedPhotonEcalIsol"), cms.InputTag("l1NonIsolatedPhotonEcalIsol")),
            theHLTOutputTypes = cms.uint32(100)
        ), 
        cms.PSet(
            PlotBounds = cms.vdouble(0.0, 10.0),
            HLTCollectionLabels = cms.InputTag("hltL1NonIsoDoublePhotonHcalIsolFilter"),
            IsoCollections = cms.VInputTag(cms.InputTag("l1IsolatedPhotonHcalIsol"), cms.InputTag("l1NonIsolatedPhotonHcalIsol")),
            theHLTOutputTypes = cms.uint32(100)
        ), 
        cms.PSet(
            PlotBounds = cms.vdouble(0.0, 10.0),
            HLTCollectionLabels = cms.InputTag("hltL1NonIsoDoublePhotonTrackIsolFilter"),
            IsoCollections = cms.VInputTag(cms.InputTag("l1IsoPhotonTrackIsol"), cms.InputTag("l1NonIsoPhotonTrackIsol")),
            theHLTOutputTypes = cms.uint32(91)
        ), 
        cms.PSet(
            PlotBounds = cms.vdouble(0.0, 0.0),
            HLTCollectionLabels = cms.InputTag("hltL1NonIsoDoublePhotonDoubleEtFilter"),
            IsoCollections = cms.VInputTag(cms.InputTag("none")),
            theHLTOutputTypes = cms.uint32(91)
        )),
    disableROOToutput = cms.untracked.bool(True),
    PtMax = cms.untracked.double(200.0)
)


