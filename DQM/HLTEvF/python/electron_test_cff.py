import FWCore.ParameterSet.Config as cms

sourcePlots = cms.EDAnalyzer("HLTMonElectronSource",
    outputFile = cms.untracked.string('./ElectronDQM.root'),
    MonitorDaemon = cms.untracked.bool(True),
    reqNum = cms.uint32(1),
    DaqMonitorBEInterface = cms.untracked.bool(True),
    filters = cms.VPSet(
        cms.PSet(
            PlotBounds = cms.vdouble(0.0, 0.0),
            HLTCollectionLabels = cms.InputTag("hltL1NonIsoHLTNonIsoSingleElectronEt10EtFilter","","HLT"),
            IsoCollections = cms.VInputTag(cms.InputTag("none")),
            theHLTOutputTypes = cms.uint32(100)
        ), 
        cms.PSet(
            PlotBounds = cms.vdouble(0.0, 10.0),
            HLTCollectionLabels = cms.InputTag("hltL1NonHLTnonIsoIsoSingleElectronEt10HcalIsolFilter","","HLT"),
            IsoCollections = cms.VInputTag(cms.InputTag("hltL1IsolatedElectronHcalIsol","","HLT"), cms.InputTag("hltL1NonIsolatedElectronHcalIsol","","HLT")),
            theHLTOutputTypes = cms.uint32(100)
        ), 
        cms.PSet(
            PlotBounds = cms.vdouble(0.0, 0.0),
            HLTCollectionLabels = cms.InputTag("hltL1NonIsoHLTNonIsoSingleElectronEt10PixelMatchFilter","","HLT"),
            IsoCollections = cms.VInputTag(cms.InputTag("none")),
            theHLTOutputTypes = cms.uint32(100)
        ), 
        cms.PSet(
            PlotBounds = cms.vdouble(0.0, 0.0),
            HLTCollectionLabels = cms.InputTag("hltL1NonIsoHLTnonIsoSingleElectronEt10HOneOEMinusOneOPFilter","","HLT"),
            IsoCollections = cms.VInputTag(cms.InputTag("none")),
            theHLTOutputTypes = cms.uint32(92)
        ), 
        cms.PSet(
            PlotBounds = cms.vdouble(0.0, 10.0),
            HLTCollectionLabels = cms.InputTag("hltL1NonIsoHLTNonIsoSingleElectronEt10TrackIsolFilter","","HLT"),
            IsoCollections = cms.VInputTag(cms.InputTag("hltL1IsoStartUpElectronTrackIsol","","HLT"), cms.InputTag("hltL1NonIsoStartupElectronTrackIsol","","HLT")),
            theHLTOutputTypes = cms.uint32(92)
        )),
    disableROOToutput = cms.untracked.bool(True),
    PtMax = cms.untracked.double(200.0)
)

clientPlots = cms.EDAnalyzer("HLTMonElectronClient",
    outputFile = cms.untracked.string('./ElectronDQM.root'),
    MonitorDaemon = cms.untracked.bool(True),
    SourceTag = cms.InputTag("sourcePlots"),
    DaqMonitorBEInterface = cms.untracked.bool(True),
    disableROOToutput = cms.untracked.bool(True),
    theHLTCollectionLabels = cms.VInputTag(
        cms.InputTag("hltL1NonHLTnonIsoIsoSingleElectronEt10HcalIsolFilter"),
        cms.InputTag("hltL1NonIsoHLTNonIsoSingleElectronEt10PixelMatchFilter"),
        cms.InputTag("hltL1NonIsoHLTnonIsoSingleElectronEt10HOneOEMinusOneOPFilter"),
        cms.InputTag("hltL1NonIsoHLTNonIsoSingleElectronEt10TrackIsolFilter")
    )
)

egammaDQMpath = cms.Path(sourcePlots * clientPlots)

