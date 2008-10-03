import FWCore.ParameterSet.Config as cms

doubleElectronCE = cms.EDFilter("EmCheckEfficiency",
    genEtaAcc = cms.double(2.5),
    doOffline = cms.bool(True),
    doMC = cms.bool(True),
    genEtAcc = cms.double(2.0),
    reqNum = cms.uint32(2),
    filters = cms.VPSet(cms.PSet(
        PlotBounds = cms.vdouble(0.0, 0.0),
        HLTCollectionLabels = cms.InputTag("hltL1sDoubleEgamma","","myHLT"),
        IsoCollections = cms.VInputTag(cms.InputTag("none")),
        theHLTOutputTypes = cms.uint32(83)
    ), 
        cms.PSet(
            PlotBounds = cms.vdouble(0.0, 0.0),
            HLTCollectionLabels = cms.InputTag("hltL1IsoDoubleElectronL1MatchFilterRegional","","myHLT"),
            IsoCollections = cms.VInputTag(cms.InputTag("none")),
            theHLTOutputTypes = cms.uint32(100)
        ), 
        cms.PSet(
            PlotBounds = cms.vdouble(0.0, 0.0),
            HLTCollectionLabels = cms.InputTag("hltL1IsoDoubleElectronEtFilter","","myHLT"),
            IsoCollections = cms.VInputTag(cms.InputTag("none")),
            theHLTOutputTypes = cms.uint32(100)
        ), 
        cms.PSet(
            PlotBounds = cms.vdouble(0.0, 10.0),
            HLTCollectionLabels = cms.InputTag("hltL1IsoDoubleElectronHcalIsolFilter","","myHLT"),
            IsoCollections = cms.VInputTag(cms.InputTag("l1IsolatedElectronHcalIsol","","myHLT")),
            theHLTOutputTypes = cms.uint32(100)
        ), 
        cms.PSet(
            PlotBounds = cms.vdouble(0.0, 0.0),
            HLTCollectionLabels = cms.InputTag("hltL1IsoDoubleElectronPixelMatchFilter","","myHLT"),
            IsoCollections = cms.VInputTag(cms.InputTag("none")),
            theHLTOutputTypes = cms.uint32(100)
        ), 
        cms.PSet(
            PlotBounds = cms.vdouble(0.0, 0.0),
            HLTCollectionLabels = cms.InputTag("hltL1IsoDoubleElectronEoverpFilter","","myHLT"),
            IsoCollections = cms.VInputTag(cms.InputTag("none")),
            theHLTOutputTypes = cms.uint32(92)
        ), 
        cms.PSet(
            PlotBounds = cms.vdouble(0.0, 10.0),
            HLTCollectionLabels = cms.InputTag("hltL1IsoDoubleElectronTrackIsolFilter","","myHLT"),
            IsoCollections = cms.VInputTag(cms.InputTag("l1IsoElectronTrackIsol","","myHLT")),
            theHLTOutputTypes = cms.uint32(92)
        )),
    PtMax = cms.untracked.double(200.0),
    pdgGen = cms.int32(11)
)


