import FWCore.ParameterSet.Config as cms

doubleEle5SWL1RDQM = cms.EDAnalyzer("EmDQM",
    genEtaAcc = cms.double(2.5),
    genEtAcc = cms.double(2.0),
    reqNum = cms.uint32(2),
    filters = cms.VPSet(cms.PSet(
        PlotBounds = cms.vdouble(0.0, 0.0),
        HLTCollectionLabels = cms.InputTag("hltL1sRelaxedDoubleEgammaEt5","","HLT"),
        IsoCollections = cms.VInputTag(cms.InputTag("none")),
        theHLTOutputTypes = cms.int32(-82)
    ), 
        cms.PSet(
            PlotBounds = cms.vdouble(0.0, 0.0),
            HLTCollectionLabels = cms.InputTag("hltL1NonIsoHLTNonIsoDoubleElectronEt5L1MatchFilterRegional","","HLT"),
            IsoCollections = cms.VInputTag(cms.InputTag("none")),
            theHLTOutputTypes = cms.int32(92)
        ), 
        cms.PSet(
            PlotBounds = cms.vdouble(0.0, 0.0),
            HLTCollectionLabels = cms.InputTag("hltL1NonIsoHLTNonIsoDoubleElectronEt5EtFilter","","HLT"),
            IsoCollections = cms.VInputTag(cms.InputTag("none")),
            theHLTOutputTypes = cms.int32(92)
        ), 
        cms.PSet(
            PlotBounds = cms.vdouble(0.0, 10.0),
            HLTCollectionLabels = cms.InputTag("hltL1NonHLTnonIsoIsoDoubleElectronEt5HcalIsolFilter","","HLT"),
            IsoCollections = cms.VInputTag(cms.InputTag("hltL1IsolatedElectronHcalIsol","","HLT"), cms.InputTag("hltL1NonIsolatedElectronHcalIsol","","HLT")),
            theHLTOutputTypes = cms.int32(92)
        ), 
        cms.PSet(
            PlotBounds = cms.vdouble(0.0, 0.0),
            HLTCollectionLabels = cms.InputTag("hltL1NonIsoHLTNonIsoDoubleElectronEt5PixelMatchFilter","","HLT"),
            IsoCollections = cms.VInputTag(cms.InputTag("none")),
            theHLTOutputTypes = cms.int32(92)
        ), 
        cms.PSet(
            PlotBounds = cms.vdouble(0.0, 0.0),
            HLTCollectionLabels = cms.InputTag("hltL1NonIsoHLTnonIsoDoubleElectronEt5HOneOEMinusOneOPFilter","","HLT"),
            IsoCollections = cms.VInputTag(cms.InputTag("none")),
            theHLTOutputTypes = cms.int32(82)
        ), 
        cms.PSet(
            PlotBounds = cms.vdouble(0.0, 10.0),
            HLTCollectionLabels = cms.InputTag("hltL1NonIsoHLTNonIsoDoubleElectronEt5TrackIsolFilter","","HLT"),
            IsoCollections = cms.VInputTag(cms.InputTag("hltL1IsoStartUpElectronTrackIsol","","HLT"), cms.InputTag("hltL1NonIsoStartupElectronTrackIsol","","HLT")),
            theHLTOutputTypes = cms.int32(82)
        )),
    PtMax = cms.untracked.double(200.0),
    pdgGen = cms.int32(11)
)



