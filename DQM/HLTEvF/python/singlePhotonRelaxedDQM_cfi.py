import FWCore.ParameterSet.Config as cms

singlePhotonRelaxedDQM = cms.EDFilter("HLTMonElectron",
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
            HLTCollectionLabels = cms.InputTag("hltL1NonIsoSinglePhotonL1MatchFilterRegional::HLT"),
            IsoCollections = cms.VInputTag(cms.InputTag("none")),
            theHLTOutputTypes = cms.uint32(100)
        ), 
        cms.PSet(
            PlotBounds = cms.vdouble(0.0, 0.0),
            HLTCollectionLabels = cms.InputTag("hltL1NonIsoSinglePhotonEtFilter"::HLT),
            IsoCollections = cms.VInputTag(cms.InputTag("none")),
            theHLTOutputTypes = cms.uint32(100)
        ), 
        cms.PSet(
            PlotBounds = cms.vdouble(0.0, 10.0),
            HLTCollectionLabels = cms.InputTag("hltL1NonIsoSinglePhotonEcalIsolFilter::HLT"),
            IsoCollections = cms.VInputTag(cms.InputTag("hltL1IsolatedPhotonEcalIsol::HLT"), cms.InputTag("hltL1NonIsolatedPhotonEcalIsol::HLT")),
            theHLTOutputTypes = cms.uint32(100)
        ), 
        cms.PSet(
            PlotBounds = cms.vdouble(0.0, 10.0),
            HLTCollectionLabels = cms.InputTag("hltL1NonIsoSinglePhotonHcalIsolFilter::HLT"),
            IsoCollections = cms.VInputTag(cms.InputTag("hltL1IsolatedPhotonHcalIsol::HLT"), cms.InputTag("hltL1NonIsolatedPhotonHcalIsol::HLT")),
            theHLTOutputTypes = cms.uint32(100)
        ), 
        cms.PSet(
            PlotBounds = cms.vdouble(0.0, 10.0),
            HLTCollectionLabels = cms.InputTag("hltL1NonIsoSinglePhotonTrackIsolFilter::HLT"),
            IsoCollections = cms.VInputTag(cms.InputTag("hltL1IsoPhotonTrackIsol::HLT"), cms.InputTag("hltL1NonIsoPhotonTrackIsol::HLT")),
            theHLTOutputTypes = cms.uint32(91)
        )),
    disableROOToutput = cms.untracked.bool(True),
    PtMax = cms.untracked.double(200.0)
)


