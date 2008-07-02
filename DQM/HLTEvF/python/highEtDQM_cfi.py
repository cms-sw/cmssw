import FWCore.ParameterSet.Config as cms

highEtDQM = cms.EDFilter("HLTMonElectron",
    outputFile = cms.untracked.string('./L1TDQM.root'),
    MonitorDaemon = cms.untracked.bool(True),
    reqNum = cms.uint32(1),
    DaqMonitorBEInterface = cms.untracked.bool(True),
    filters = cms.VPSet(cms.PSet(
        PlotBounds = cms.vdouble(0.0, 0.0),
        HLTCollectionLabels = cms.InputTag("hltL1seedRelaxedSingleEgamma::HLT"),
        IsoCollections = cms.VInputTag(cms.InputTag("none")),
        theHLTOutputTypes = cms.uint32(82)
    ), 
        cms.PSet(
            PlotBounds = cms.vdouble(0.0, 0.0),
            HLTCollectionLabels = cms.InputTag("hltL1NonIsoSingleEMHighEtL1MatchFilterRegional::HLT"),
            IsoCollections = cms.VInputTag(cms.InputTag("none")),
            theHLTOutputTypes = cms.uint32(100)
        ), 
        cms.PSet(
            PlotBounds = cms.vdouble(0.0, 0.0),
            HLTCollectionLabels = cms.InputTag("hltL1NonIsoSinglePhotonEMHighEtEtFilter::HLT"),
            IsoCollections = cms.VInputTag(cms.InputTag("none")),
            theHLTOutputTypes = cms.uint32(100)
        ), 
        cms.PSet(
            PlotBounds = cms.vdouble(0.0, 200.0),
            HLTCollectionLabels = cms.InputTag("hltL1NonIsoSingleEMHighEtEcalIsolFilter::HLT"),
            IsoCollections = cms.VInputTag(cms.InputTag("hltL1IsolatedPhotonEcalIsol::HLT"), cms.InputTag("hltL1NonIsolatedPhotonEcalIsol::HLT")),
            theHLTOutputTypes = cms.uint32(100)
        ), 
        cms.PSet(
            PlotBounds = cms.vdouble(0.0, 200.0),
            HLTCollectionLabels = cms.InputTag("hltL1NonIsoSingleEMHighEtHOEFilter::HLT"),
            IsoCollections = cms.VInputTag(cms.InputTag("hltL1NonIsolatedElectronHcalIsol::HLT"), cms.InputTag("hltL1IsolatedElectronHcalIsol::HLT")),
            theHLTOutputTypes = cms.uint32(100)
        ), 
        cms.PSet(
            PlotBounds = cms.vdouble(0.0, 200.0),
            HLTCollectionLabels = cms.InputTag("hltL1NonIsoSingleEMHighEtHcalDBCFilter::HLT"),
            IsoCollections = cms.VInputTag(cms.InputTag("hltHcalDoubleCone::HLT"), cms.InputTag("hltL1NonIsoEMHcalDoubleCone::HLT")),
            theHLTOutputTypes = cms.uint32(100)
        ), 
        cms.PSet(
            PlotBounds = cms.vdouble(0.0, 10.0),
            HLTCollectionLabels = cms.InputTag("hltL1NonIsoSingleEMHighEtTrackIsolFilter::HLT"),
            IsoCollections = cms.VInputTag(cms.InputTag("hltL1IsoPhotonTrackIsol::HLT"), cms.InputTag("hltL1NonIsoPhotonTrackIsol::HLT")),
            theHLTOutputTypes = cms.uint32(91)
        )),
    disableROOToutput = cms.untracked.bool(True),
    PtMax = cms.untracked.double(1000.0)
)


