import FWCore.ParameterSet.Config as cms

doubleElectronRelaxedCE = cms.EDFilter("EmCheckEfficiency",
    genEtaAcc = cms.double(2.5),
    doOffline = cms.bool(True),
    doMC = cms.bool(True),
    genEtAcc = cms.double(2.0),
    reqNum = cms.uint32(2),
    filters = cms.VPSet(cms.PSet(
        PlotBounds = cms.vdouble(0.0, 0.0),
        HLTCollectionLabels = cms.InputTag("hltL1sRelaxedDoubleEgamma","","myHLT"),
        IsoCollections = cms.VInputTag(cms.InputTag("none")),
        theHLTOutputTypes = cms.uint32(82)
    ), 
        cms.PSet(
            PlotBounds = cms.vdouble(0.0, 0.0),
            HLTCollectionLabels = cms.InputTag("hltL1NonIsoDoubleElectronL1MatchFilterRegional","","myHLT"),
            IsoCollections = cms.VInputTag(cms.InputTag("none")),
            theHLTOutputTypes = cms.uint32(100)
        ), 
        cms.PSet(
            PlotBounds = cms.vdouble(0.0, 0.0),
            HLTCollectionLabels = cms.InputTag("hltL1NonIsoDoubleElectronEtFilter","","myHLT"),
            IsoCollections = cms.VInputTag(cms.InputTag("none")),
            theHLTOutputTypes = cms.uint32(100)
        ), 
        cms.PSet(
            PlotBounds = cms.vdouble(0.0, 10.0),
            HLTCollectionLabels = cms.InputTag("hltL1NonIsoDoubleElectronHcalIsolFilter","","myHLT"),
            IsoCollections = cms.VInputTag(cms.InputTag("l1IsolatedElectronHcalIsol","","myHLT"), cms.InputTag("l1NonIsolatedElectronHcalIsol","","HLT")),
            theHLTOutputTypes = cms.uint32(100)
        ), 
        cms.PSet(
            PlotBounds = cms.vdouble(0.0, 0.0),
            HLTCollectionLabels = cms.InputTag("hltL1NonIsoDoubleElectronPixelMatchFilter","","myHLT"),
            IsoCollections = cms.VInputTag(cms.InputTag("none")),
            theHLTOutputTypes = cms.uint32(100)
        ), 
        cms.PSet(
            PlotBounds = cms.vdouble(0.0, 0.0),
            HLTCollectionLabels = cms.InputTag("hltL1NonIsoDoubleElectronEoverpFilter","","myHLT"),
            IsoCollections = cms.VInputTag(cms.InputTag("none")),
            theHLTOutputTypes = cms.uint32(92)
        ), 
        cms.PSet(
            PlotBounds = cms.vdouble(0.0, 10.0),
            HLTCollectionLabels = cms.InputTag("hltL1NonIsoDoubleElectronTrackIsolFilter","","myHLT"),
            IsoCollections = cms.VInputTag(cms.InputTag("l1IsoElectronTrackIsol","","myHLT"), cms.InputTag("l1NonIsoElectronTrackIsol","","HLT")),
            theHLTOutputTypes = cms.uint32(92)
        )),
    PtMax = cms.untracked.double(200.0),
    pdgGen = cms.int32(11)
)


