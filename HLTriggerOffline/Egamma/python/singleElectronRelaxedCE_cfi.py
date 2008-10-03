import FWCore.ParameterSet.Config as cms

singleElectronRelaxedCE = cms.EDFilter("EmCheckEfficiency",
    genEtaAcc = cms.double(2.5),
    doOffline = cms.bool(True),
    doMC = cms.bool(True),
    genEtAcc = cms.double(2.0),
    reqNum = cms.uint32(1),
    filters = cms.VPSet(cms.PSet(
        PlotBounds = cms.vdouble(0.0, 0.0),
        HLTCollectionLabels = cms.InputTag("hltL1sRelaxedSingleEgamma","","myHLT"),
        IsoCollections = cms.VInputTag(cms.InputTag("none")),
        theHLTOutputTypes = cms.uint32(82)
    ), 
        cms.PSet(
            PlotBounds = cms.vdouble(0.0, 0.0),
            HLTCollectionLabels = cms.InputTag("hltL1NonIsoSingleElectronL1MatchFilterRegional","","myHLT"),
            IsoCollections = cms.VInputTag(cms.InputTag("none")),
            theHLTOutputTypes = cms.uint32(100)
        ), 
        cms.PSet(
            PlotBounds = cms.vdouble(0.0, 0.0),
            HLTCollectionLabels = cms.InputTag("hltL1NonIsoSingleElectronEtFilter","","myHLT"),
            IsoCollections = cms.VInputTag(cms.InputTag("none")),
            theHLTOutputTypes = cms.uint32(100)
        ), 
        cms.PSet(
            PlotBounds = cms.vdouble(0.0, 10.0),
            HLTCollectionLabels = cms.InputTag("hltL1NonIsoSingleElectronHcalIsolFilter","","myHLT"),
            IsoCollections = cms.VInputTag(cms.InputTag("l1IsolatedElectronHcalIsol","","myHLT"), cms.InputTag("l1NonIsolatedElectronHcalIsol","","HLT")),
            theHLTOutputTypes = cms.uint32(100)
        ), 
        cms.PSet(
            PlotBounds = cms.vdouble(0.0, 0.0),
            HLTCollectionLabels = cms.InputTag("hltL1NonIsoSingleElectronPixelMatchFilter","","myHLT"),
            IsoCollections = cms.VInputTag(cms.InputTag("none")),
            theHLTOutputTypes = cms.uint32(100)
        ), 
        cms.PSet(
            PlotBounds = cms.vdouble(0.0, 0.0),
            HLTCollectionLabels = cms.InputTag("hltL1NonIsoSingleElectronHOneOEMinusOneOPFilter","","myHLT"),
            IsoCollections = cms.VInputTag(cms.InputTag("none")),
            theHLTOutputTypes = cms.uint32(92)
        ), 
        cms.PSet(
            PlotBounds = cms.vdouble(0.0, 0.0),
            HLTCollectionLabels = cms.InputTag("hltL1NonIsoSingleElectronDetaDphiFilter","","myHLT"),
            IsoCollections = cms.VInputTag(cms.InputTag("none")),
            theHLTOutputTypes = cms.uint32(92)
        ), 
        cms.PSet(
            PlotBounds = cms.vdouble(0.0, 10.0),
            HLTCollectionLabels = cms.InputTag("hltL1NonIsoSingleElectronTrackIsolFilter","","myHLT"),
            IsoCollections = cms.VInputTag(cms.InputTag("l1IsoElectronTrackIsol","","myHLT"), cms.InputTag("l1NonIsoElectronTrackIsol","","HLT")),
            theHLTOutputTypes = cms.uint32(92)
        )),
    PtMax = cms.untracked.double(200.0),
    pdgGen = cms.int32(11)
)


