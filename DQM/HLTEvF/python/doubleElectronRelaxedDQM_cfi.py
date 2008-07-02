import FWCore.ParameterSet.Config as cms

doubleElectronRelaxedDQM = cms.EDFilter("HLTMonElectron",
    outputFile = cms.untracked.string('./L1TDQM.root'),
    MonitorDaemon = cms.untracked.bool(True),
    reqNum = cms.uint32(2),
    DaqMonitorBEInterface = cms.untracked.bool(True),
    filters = cms.VPSet(cms.PSet(
        PlotBounds = cms.vdouble(0.0, 0.0),
        HLTCollectionLabels = cms.InputTag("hltL1seedRelaxedDoubleEgamma::HLT"),
        IsoCollections = cms.VInputTag(cms.InputTag("none")),
        theHLTOutputTypes = cms.uint32(82)
    ), 
        cms.PSet(
            PlotBounds = cms.vdouble(0.0, 0.0),
            HLTCollectionLabels = cms.InputTag("hltL1NonIsoDoubleElectronL1MatchFilterRegional::HLT"),
            IsoCollections = cms.VInputTag(cms.InputTag("none")),
            theHLTOutputTypes = cms.uint32(100)
        ), 
        cms.PSet(
            PlotBounds = cms.vdouble(0.0, 0.0),
            HLTCollectionLabels = cms.InputTag("hltL1NonIsoDoubleElectronEtFilter::HLT"),
            IsoCollections = cms.VInputTag(cms.InputTag("none")),
            theHLTOutputTypes = cms.uint32(100)
        ), 
        cms.PSet(
            PlotBounds = cms.vdouble(0.0, 10.0),
            HLTCollectionLabels = cms.InputTag("hltL1NonIsoDoubleElectronHcalIsolFilter::HLT"),
            IsoCollections = cms.VInputTag(cms.InputTag("hltL1IsolatedElectronHcalIsol::HLT"), cms.InputTag("hltL1NonIsolatedElectronHcalIsol::HLT")),
            theHLTOutputTypes = cms.uint32(100)
        ), 
        cms.PSet(
            PlotBounds = cms.vdouble(0.0, 0.0),
            HLTCollectionLabels = cms.InputTag("hltL1NonIsoDoubleElectronPixelMatchFilter::HLT"),
            IsoCollections = cms.VInputTag(cms.InputTag("none")),
            theHLTOutputTypes = cms.uint32(100)
        ), 
        cms.PSet(
            PlotBounds = cms.vdouble(0.0, 0.0),
            HLTCollectionLabels = cms.InputTag("hltL1NonIsoDoubleElectronEoverpFilter::HLT"),
            IsoCollections = cms.VInputTag(cms.InputTag("none")),
            theHLTOutputTypes = cms.uint32(92)
        ), 
        cms.PSet(
            PlotBounds = cms.vdouble(0.0, 10.0),
            HLTCollectionLabels = cms.InputTag("hltL1NonIsoDoubleElectronTrackIsolFilter::HLT"),
            IsoCollections = cms.VInputTag(cms.InputTag("hltL1IsoElectronTrackIsol::HLT"), cms.InputTag("hltL1NonIsoElectronTrackIsol::HLT")),
            theHLTOutputTypes = cms.uint32(92)
        )),
    disableROOToutput = cms.untracked.bool(True),
    PtMax = cms.untracked.double(200.0)
)


