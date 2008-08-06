import FWCore.ParameterSet.Config as cms

hltMonE = cms.EDAnalyzer("HLTMon",
    outputFile = cms.untracked.string('./L1TDQM.root'),
    verbose = cms.untracked.bool(False),
    MonitorDaemon = cms.untracked.bool(True),
    reqNum = cms.uint32(1),
    DaqMonitorBEInterface = cms.untracked.bool(True),
    filters = cms.VPSet(cms.PSet(
        PlotBounds = cms.vdouble(0.0, 0.0),
        HLTCollectionLabels = cms.InputTag("hltL1seedRelaxedSingleEt8","","FU"),
        IsoCollections = cms.VInputTag(cms.InputTag("none")),
        theHLTOutputTypes = cms.uint32(83)
    ), 
        cms.PSet(
            PlotBounds = cms.vdouble(0.0, 0.0),
            HLTCollectionLabels = cms.InputTag("hltL1NonIsoHLTNonIsoSingleElectronLWEt12L1MatchFilterRegional","","FU"),
            IsoCollections = cms.VInputTag(cms.InputTag("none")),
            theHLTOutputTypes = cms.uint32(100)
        ), 
        cms.PSet(
            PlotBounds = cms.vdouble(0.0, 0.0),
            HLTCollectionLabels = cms.InputTag("hltL1NonIsoHLTNonIsoSingleElectronLWEt12EtFilter","","FU"),
            IsoCollections = cms.VInputTag(cms.InputTag("none")),
            theHLTOutputTypes = cms.uint32(100)
        ), 
        cms.PSet(
            PlotBounds = cms.vdouble(0.0, 10.0),
            HLTCollectionLabels = cms.InputTag("hltL1NonIsoHLTNonIsoSingleElectronLWEt12HcalIsolFilter","","FU"),
            IsoCollections = cms.VInputTag(cms.InputTag("hltL1NonIsolatedElectronHcalIsol")),
            theHLTOutputTypes = cms.uint32(100)
        ), 
        cms.PSet(
            PlotBounds = cms.vdouble(0.0, 0.0),
            HLTCollectionLabels = cms.InputTag("hltL1NonIsoHLTNonIsoSingleElectronLWEt12PixelMatchFilter","","FU"),
            IsoCollections = cms.VInputTag(cms.InputTag("none")),
            theHLTOutputTypes = cms.uint32(100)
        ), 
        cms.PSet(
            PlotBounds = cms.vdouble(0.0, 0.0),
            HLTCollectionLabels = cms.InputTag("hltL1NonIsoHLTNonIsoSingleElectronLWEt12HOneOEMinusOneOPFilter","","FU"),
            IsoCollections = cms.VInputTag(cms.InputTag("none")),
            theHLTOutputTypes = cms.uint32(100)
        ), 
        cms.PSet(
            PlotBounds = cms.vdouble(0.0, 10.0),
            HLTCollectionLabels = cms.InputTag("hltL1NonIsoHLTNonIsoSingleElectronLWEt12TrackIsolFilter","","FU"),
            IsoCollections = cms.VInputTag(cms.InputTag("hltL1NonIsoLargeWindowElectronTrackIsol")),
            theHLTOutputTypes = cms.uint32(92)
        ), 
        cms.PSet(
            PlotBounds = cms.vdouble(0.0, 0.0),
            HLTCollectionLabels = cms.InputTag("hltL1seedDoubleEgamma","","FU"),
            IsoCollections = cms.VInputTag(cms.InputTag("none")),
            theHLTOutputTypes = cms.uint32(83)
        ), 
        cms.PSet(
            PlotBounds = cms.vdouble(0.0, 0.0),
            HLTCollectionLabels = cms.InputTag("hltL1IsoDoubleElectronL1MatchFilterRegional","","FU"),
            IsoCollections = cms.VInputTag(cms.InputTag("none")),
            theHLTOutputTypes = cms.uint32(100)
        ), 
        cms.PSet(
            PlotBounds = cms.vdouble(0.0, 0.0),
            HLTCollectionLabels = cms.InputTag("hltL1IsoDoubleElectronEtFilter","","FU"),
            IsoCollections = cms.VInputTag(cms.InputTag("none")),
            theHLTOutputTypes = cms.uint32(100)
        ), 
        cms.PSet(
            PlotBounds = cms.vdouble(0.0, 10.0),
            HLTCollectionLabels = cms.InputTag("hltL1IsoDoubleElectronHcalIsolFilter","","FU"),
            IsoCollections = cms.VInputTag(cms.InputTag("hltL1IsolatedElectronHcalIsol","","FU")),
            theHLTOutputTypes = cms.uint32(100)
        ), 
        cms.PSet(
            PlotBounds = cms.vdouble(0.0, 0.0),
            HLTCollectionLabels = cms.InputTag("hltL1IsoDoubleElectronPixelMatchFilter","","FU"),
            IsoCollections = cms.VInputTag(cms.InputTag("none")),
            theHLTOutputTypes = cms.uint32(100)
        ), 
        cms.PSet(
            PlotBounds = cms.vdouble(0.0, 0.0),
            HLTCollectionLabels = cms.InputTag("hltL1IsoDoubleElectronEoverpFilter","","FU"),
            IsoCollections = cms.VInputTag(cms.InputTag("none")),
            theHLTOutputTypes = cms.uint32(92)
        ), 
        cms.PSet(
            PlotBounds = cms.vdouble(0.0, 10.0),
            HLTCollectionLabels = cms.InputTag("hltL1IsoDoubleElectronTrackIsolFilter","","FU"),
            IsoCollections = cms.VInputTag(cms.InputTag("hltL1IsoElectronTrackIsol","","FU")),
            theHLTOutputTypes = cms.uint32(92)
        ), 
        cms.PSet(
            PlotBounds = cms.vdouble(0.0, 0.0),
            HLTCollectionLabels = cms.InputTag("hltL1seedRelaxedDoubleEgamma","","FU"),
            IsoCollections = cms.VInputTag(cms.InputTag("none")),
            theHLTOutputTypes = cms.uint32(82)
        ), 
        cms.PSet(
            PlotBounds = cms.vdouble(0.0, 0.0),
            HLTCollectionLabels = cms.InputTag("hltL1NonIsoDoubleElectronL1MatchFilterRegional","","FU"),
            IsoCollections = cms.VInputTag(cms.InputTag("none")),
            theHLTOutputTypes = cms.uint32(100)
        ), 
        cms.PSet(
            PlotBounds = cms.vdouble(0.0, 0.0),
            HLTCollectionLabels = cms.InputTag("hltL1NonIsoDoubleElectronEtFilter","","FU"),
            IsoCollections = cms.VInputTag(cms.InputTag("none")),
            theHLTOutputTypes = cms.uint32(100)
        ), 
        cms.PSet(
            PlotBounds = cms.vdouble(0.0, 10.0),
            HLTCollectionLabels = cms.InputTag("hltL1NonIsoDoubleElectronHcalIsolFilter","","FU"),
            IsoCollections = cms.VInputTag(cms.InputTag("hltL1IsolatedElectronHcalIsol","","FU"), cms.InputTag("hltL1NonIsolatedElectronHcalIsol","","FU")),
            theHLTOutputTypes = cms.uint32(100)
        ), 
        cms.PSet(
            PlotBounds = cms.vdouble(0.0, 0.0),
            HLTCollectionLabels = cms.InputTag("hltL1NonIsoDoubleElectronPixelMatchFilter","","FU"),
            IsoCollections = cms.VInputTag(cms.InputTag("none")),
            theHLTOutputTypes = cms.uint32(100)
        ), 
        cms.PSet(
            PlotBounds = cms.vdouble(0.0, 0.0),
            HLTCollectionLabels = cms.InputTag("hltL1NonIsoDoubleElectronEoverpFilter","","FU"),
            IsoCollections = cms.VInputTag(cms.InputTag("none")),
            theHLTOutputTypes = cms.uint32(92)
        ), 
        cms.PSet(
            PlotBounds = cms.vdouble(0.0, 10.0),
            HLTCollectionLabels = cms.InputTag("hltL1NonIsoDoubleElectronTrackIsolFilter","","FU"),
            IsoCollections = cms.VInputTag(cms.InputTag("hltL1IsoElectronTrackIsol","","FU"), cms.InputTag("hltL1NonIsoElectronTrackIsol","","FU")),
            theHLTOutputTypes = cms.uint32(92)
        ), 
        cms.PSet(
            PlotBounds = cms.vdouble(0.0, 0.0),
            HLTCollectionLabels = cms.InputTag("hltL1seedDoubleEgamma","","FU"),
            IsoCollections = cms.VInputTag(cms.InputTag("none")),
            theHLTOutputTypes = cms.uint32(83)
        ), 
        cms.PSet(
            PlotBounds = cms.vdouble(0.0, 0.0),
            HLTCollectionLabels = cms.InputTag("hltL1IsoDoublePhotonL1MatchFilterRegional","","FU"),
            IsoCollections = cms.VInputTag(cms.InputTag("none")),
            theHLTOutputTypes = cms.uint32(100)
        ), 
        cms.PSet(
            PlotBounds = cms.vdouble(0.0, 0.0),
            HLTCollectionLabels = cms.InputTag("hltL1IsoDoublePhotonEtFilter","","FU"),
            IsoCollections = cms.VInputTag(cms.InputTag("none")),
            theHLTOutputTypes = cms.uint32(100)
        ), 
        cms.PSet(
            PlotBounds = cms.vdouble(0.0, 10.0),
            HLTCollectionLabels = cms.InputTag("hltL1IsoDoublePhotonEcalIsolFilter","","FU"),
            IsoCollections = cms.VInputTag(cms.InputTag("hltL1IsolatedPhotonEcalIsol","","FU")),
            theHLTOutputTypes = cms.uint32(100)
        ), 
        cms.PSet(
            PlotBounds = cms.vdouble(0.0, 10.0),
            HLTCollectionLabels = cms.InputTag("hltL1IsoDoublePhotonHcalIsolFilter","","FU"),
            IsoCollections = cms.VInputTag(cms.InputTag("hltL1IsolatedPhotonHcalIsol","","FU")),
            theHLTOutputTypes = cms.uint32(100)
        ), 
        cms.PSet(
            PlotBounds = cms.vdouble(0.0, 10.0),
            HLTCollectionLabels = cms.InputTag("hltL1IsoDoublePhotonTrackIsolFilter","","FU"),
            IsoCollections = cms.VInputTag(cms.InputTag("hltL1IsoPhotonTrackIsol","","FU")),
            theHLTOutputTypes = cms.uint32(91)
        ), 
        cms.PSet(
            PlotBounds = cms.vdouble(0.0, 0.0),
            HLTCollectionLabels = cms.InputTag("hltL1seedRelaxedDoubleEgamma","","FU"),
            IsoCollections = cms.VInputTag(cms.InputTag("none")),
            theHLTOutputTypes = cms.uint32(82)
        ), 
        cms.PSet(
            PlotBounds = cms.vdouble(0.0, 0.0),
            HLTCollectionLabels = cms.InputTag("hltL1NonIsoDoublePhotonL1MatchFilterRegional","","FU"),
            IsoCollections = cms.VInputTag(cms.InputTag("none")),
            theHLTOutputTypes = cms.uint32(100)
        ), 
        cms.PSet(
            PlotBounds = cms.vdouble(0.0, 0.0),
            HLTCollectionLabels = cms.InputTag("hltL1NonIsoDoublePhotonEtFilter","","FU"),
            IsoCollections = cms.VInputTag(cms.InputTag("none")),
            theHLTOutputTypes = cms.uint32(100)
        ), 
        cms.PSet(
            PlotBounds = cms.vdouble(0.0, 10.0),
            HLTCollectionLabels = cms.InputTag("hltL1NonIsoDoublePhotonEcalIsolFilter","","FU"),
            IsoCollections = cms.VInputTag(cms.InputTag("hltL1IsolatedPhotonEcalIsol","","FU"), cms.InputTag("hltL1NonIsolatedPhotonEcalIsol","","FU")),
            theHLTOutputTypes = cms.uint32(100)
        ), 
        cms.PSet(
            PlotBounds = cms.vdouble(0.0, 10.0),
            HLTCollectionLabels = cms.InputTag("hltL1NonIsoDoublePhotonHcalIsolFilter","","FU"),
            IsoCollections = cms.VInputTag(cms.InputTag("hltL1IsolatedPhotonHcalIsol","","FU"), cms.InputTag("hltL1NonIsolatedPhotonHcalIsol","","FU")),
            theHLTOutputTypes = cms.uint32(100)
        ), 
        cms.PSet(
            PlotBounds = cms.vdouble(0.0, 10.0),
            HLTCollectionLabels = cms.InputTag("hltL1NonIsoDoublePhotonTrackIsolFilter","","FU"),
            IsoCollections = cms.VInputTag(cms.InputTag("hltL1IsoPhotonTrackIsol","","FU"), cms.InputTag("hltL1NonIsoPhotonTrackIsol","","FU")),
            theHLTOutputTypes = cms.uint32(91)
        ), 
        cms.PSet(
            PlotBounds = cms.vdouble(0.0, 0.0),
            HLTCollectionLabels = cms.InputTag("hltL1NonIsoDoublePhotonDoubleEtFilter","","FU"),
            IsoCollections = cms.VInputTag(cms.InputTag("none")),
            theHLTOutputTypes = cms.uint32(91)
        ), 
        cms.PSet(
            PlotBounds = cms.vdouble(0.0, 0.0),
            HLTCollectionLabels = cms.InputTag("hltL1seedRelaxedSingleEgamma","","FU"),
            IsoCollections = cms.VInputTag(cms.InputTag("none")),
            theHLTOutputTypes = cms.uint32(82)
        ), 
        cms.PSet(
            PlotBounds = cms.vdouble(0.0, 0.0),
            HLTCollectionLabels = cms.InputTag("hltL1NonIsoSingleEMHighEtL1MatchFilterRegional","","FU"),
            IsoCollections = cms.VInputTag(cms.InputTag("none")),
            theHLTOutputTypes = cms.uint32(100)
        ), 
        cms.PSet(
            PlotBounds = cms.vdouble(0.0, 0.0),
            HLTCollectionLabels = cms.InputTag("hltL1NonIsoSinglePhotonEMHighEtEtFilter","","FU"),
            IsoCollections = cms.VInputTag(cms.InputTag("none")),
            theHLTOutputTypes = cms.uint32(100)
        ), 
        cms.PSet(
            PlotBounds = cms.vdouble(0.0, 200.0),
            HLTCollectionLabels = cms.InputTag("hltL1NonIsoSingleEMHighEtEcalIsolFilter","","FU"),
            IsoCollections = cms.VInputTag(cms.InputTag("hltL1IsolatedPhotonEcalIsol","","FU"), cms.InputTag("hltL1NonIsolatedPhotonEcalIsol","","FU")),
            theHLTOutputTypes = cms.uint32(100)
        ), 
        cms.PSet(
            PlotBounds = cms.vdouble(0.0, 200.0),
            HLTCollectionLabels = cms.InputTag("hltL1NonIsoSingleEMHighEtHOEFilter","","FU"),
            IsoCollections = cms.VInputTag(cms.InputTag("hltL1NonIsolatedElectronHcalIsol","","FU"), cms.InputTag("hltL1IsolatedElectronHcalIsol","","FU")),
            theHLTOutputTypes = cms.uint32(100)
        ), 
        cms.PSet(
            PlotBounds = cms.vdouble(0.0, 200.0),
            HLTCollectionLabels = cms.InputTag("hltL1NonIsoSingleEMHighEtHcalDBCFilter","","FU"),
            IsoCollections = cms.VInputTag(cms.InputTag("hltHcalDoubleCone","","FU"), cms.InputTag("hltL1NonIsoEMHcalDoubleCone","","FU")),
            theHLTOutputTypes = cms.uint32(100)
        ), 
        cms.PSet(
            PlotBounds = cms.vdouble(0.0, 10.0),
            HLTCollectionLabels = cms.InputTag("hltL1NonIsoSingleEMHighEtTrackIsolFilter","","FU"),
            IsoCollections = cms.VInputTag(cms.InputTag("hltL1IsoPhotonTrackIsol","","FU"), cms.InputTag("hltL1NonIsoPhotonTrackIsol","","FU")),
            theHLTOutputTypes = cms.uint32(91)
        ), 
        cms.PSet(
            PlotBounds = cms.vdouble(0.0, 0.0),
            HLTCollectionLabels = cms.InputTag("hltL1seedSingleEgamma","","FU"),
            IsoCollections = cms.VInputTag(cms.InputTag("none")),
            theHLTOutputTypes = cms.uint32(83)
        ), 
        cms.PSet(
            PlotBounds = cms.vdouble(0.0, 0.0),
            HLTCollectionLabels = cms.InputTag("hltL1IsoSingleL1MatchFilter","","FU"),
            IsoCollections = cms.VInputTag(cms.InputTag("none")),
            theHLTOutputTypes = cms.uint32(100)
        ), 
        cms.PSet(
            PlotBounds = cms.vdouble(0.0, 0.0),
            HLTCollectionLabels = cms.InputTag("hltL1IsoSingleElectronEtFilter","","FU"),
            IsoCollections = cms.VInputTag(cms.InputTag("none")),
            theHLTOutputTypes = cms.uint32(100)
        ), 
        cms.PSet(
            PlotBounds = cms.vdouble(0.0, 10.0),
            HLTCollectionLabels = cms.InputTag("hltL1IsoSingleElectronHcalIsolFilter","","FU"),
            IsoCollections = cms.VInputTag(cms.InputTag("hltL1IsolatedElectronHcalIsol","","FU")),
            theHLTOutputTypes = cms.uint32(100)
        ), 
        cms.PSet(
            PlotBounds = cms.vdouble(0.0, 0.0),
            HLTCollectionLabels = cms.InputTag("hltL1IsoSingleElectronPixelMatchFilter","","FU"),
            IsoCollections = cms.VInputTag(cms.InputTag("none")),
            theHLTOutputTypes = cms.uint32(100)
        ), 
        cms.PSet(
            PlotBounds = cms.vdouble(0.0, 0.0),
            HLTCollectionLabels = cms.InputTag("hltL1IsoSingleElectronHOneOEMinusOneOPFilter","","FU"),
            IsoCollections = cms.VInputTag(cms.InputTag("none")),
            theHLTOutputTypes = cms.uint32(92)
        ), 
        cms.PSet(
            PlotBounds = cms.vdouble(0.0, 10.0),
            HLTCollectionLabels = cms.InputTag("hltL1IsoSingleElectronTrackIsolFilter","","FU"),
            IsoCollections = cms.VInputTag(cms.InputTag("hltL1IsoElectronTrackIsol","","FU")),
            theHLTOutputTypes = cms.uint32(92)
        ), 
        cms.PSet(
            PlotBounds = cms.vdouble(0.0, 0.0),
            HLTCollectionLabels = cms.InputTag("hltL1seedRelaxedSingleEgamma","","FU"),
            IsoCollections = cms.VInputTag(cms.InputTag("none")),
            theHLTOutputTypes = cms.uint32(82)
        ), 
        cms.PSet(
            PlotBounds = cms.vdouble(0.0, 0.0),
            HLTCollectionLabels = cms.InputTag("hltL1NonIsoSingleElectronL1MatchFilterRegional","","FU"),
            IsoCollections = cms.VInputTag(cms.InputTag("none")),
            theHLTOutputTypes = cms.uint32(100)
        ), 
        cms.PSet(
            PlotBounds = cms.vdouble(0.0, 0.0),
            HLTCollectionLabels = cms.InputTag("hltL1NonIsoSingleElectronEtFilter","","FU"),
            IsoCollections = cms.VInputTag(cms.InputTag("none")),
            theHLTOutputTypes = cms.uint32(100)
        ), 
        cms.PSet(
            PlotBounds = cms.vdouble(0.0, 10.0),
            HLTCollectionLabels = cms.InputTag("hltL1NonIsoSingleElectronHcalIsolFilter","","FU"),
            IsoCollections = cms.VInputTag(cms.InputTag("hltL1IsolatedElectronHcalIsol","","FU"), cms.InputTag("hltL1NonIsolatedElectronHcalIsol","","FU")),
            theHLTOutputTypes = cms.uint32(100)
        ), 
        cms.PSet(
            PlotBounds = cms.vdouble(0.0, 0.0),
            HLTCollectionLabels = cms.InputTag("hltL1NonIsoSingleElectronPixelMatchFilter","","FU"),
            IsoCollections = cms.VInputTag(cms.InputTag("none")),
            theHLTOutputTypes = cms.uint32(100)
        ), 
        cms.PSet(
            PlotBounds = cms.vdouble(0.0, 0.0),
            HLTCollectionLabels = cms.InputTag("hltL1NonIsoSingleElectronHOneOEMinusOneOPFilter","","FU"),
            IsoCollections = cms.VInputTag(cms.InputTag("none")),
            theHLTOutputTypes = cms.uint32(92)
        ), 
        cms.PSet(
            PlotBounds = cms.vdouble(0.0, 10.0),
            HLTCollectionLabels = cms.InputTag("hltL1NonIsoSingleElectronTrackIsolFilter","","FU"),
            IsoCollections = cms.VInputTag(cms.InputTag("hltL1IsoElectronTrackIsol","","FU"), cms.InputTag("hltL1NonIsoElectronTrackIsol","","FU")),
            theHLTOutputTypes = cms.uint32(92)
        )),
    disableROOToutput = cms.untracked.bool(True)
)


