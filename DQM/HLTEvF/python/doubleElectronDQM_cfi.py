import FWCore.ParameterSet.Config as cms

doubleElectronDQM = cms.EDFilter("HLTMonElectron",
    outputFile = cms.untracked.string('./L1TDQM.root'),
    MonitorDaemon = cms.untracked.bool(True),
    reqNum = cms.uint32(2),
    DaqMonitorBEInterface = cms.untracked.bool(True),
    filters = cms.VPSet(cms.PSet(
        PlotBounds = cms.vdouble(0.0, 0.0),
        HLTCollectionLabels = cms.InputTag("hltL1seedDoubleEgamma::HLT"),
        IsoCollections = cms.VInputTag(cms.InputTag("none")),
        theHLTOutputTypes = cms.uint32(83)
    ), 
        cms.PSet(
            PlotBounds = cms.vdouble(0.0, 0.0),
            HLTCollectionLabels = cms.InputTag("hltL1IsoDoubleElectronL1MatchFilterRegional::HLT"),
            IsoCollections = cms.VInputTag(cms.InputTag("none")),
            theHLTOutputTypes = cms.uint32(100)
        ), 
        cms.PSet(
            PlotBounds = cms.vdouble(0.0, 0.0),
            HLTCollectionLabels = cms.InputTag("hltL1IsoDoubleElectronEtFilter::HLT"),
            IsoCollections = cms.VInputTag(cms.InputTag("none")),
            theHLTOutputTypes = cms.uint32(100)
        ), 
        cms.PSet(
            PlotBounds = cms.vdouble(0.0, 10.0),
            HLTCollectionLabels = cms.InputTag("hltL1IsoDoubleElectronHcalIsolFilter::HLT"),
            IsoCollections = cms.VInputTag(cms.InputTag("hltL1IsolatedElectronHcalIsol::HLT")),
            theHLTOutputTypes = cms.uint32(100)
        ), 
        cms.PSet(
            PlotBounds = cms.vdouble(0.0, 0.0),
            HLTCollectionLabels = cms.InputTag("hltL1IsoDoubleElectronPixelMatchFilter::HLT"),
            IsoCollections = cms.VInputTag(cms.InputTag("none")),
            theHLTOutputTypes = cms.uint32(100)
        ), 
        cms.PSet(
            PlotBounds = cms.vdouble(0.0, 0.0),
            HLTCollectionLabels = cms.InputTag("hltL1IsoDoubleElectronEoverpFilter::HLT"),
            IsoCollections = cms.VInputTag(cms.InputTag("none")),
            theHLTOutputTypes = cms.uint32(92)
        ), 
        cms.PSet(
            PlotBounds = cms.vdouble(0.0, 10.0),
            HLTCollectionLabels = cms.InputTag("hltL1IsoDoubleElectronTrackIsolFilter::HLT"),
            IsoCollections = cms.VInputTag(cms.InputTag("hltL1IsoElectronTrackIsol::HLT")),
            theHLTOutputTypes = cms.uint32(92)
        )),
    disableROOToutput = cms.untracked.bool(True),
    PtMax = cms.untracked.double(200.0)
)


