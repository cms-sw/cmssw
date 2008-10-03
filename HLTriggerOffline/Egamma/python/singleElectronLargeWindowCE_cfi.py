import FWCore.ParameterSet.Config as cms

singleElectronLargeWindowCE = cms.EDFilter("EmCheckEfficiency",
    genEtaAcc = cms.double(2.5),
    doOffline = cms.bool(True),
    doMC = cms.bool(True),
    genEtAcc = cms.double(2.0),
    reqNum = cms.uint32(1),
    filters = cms.VPSet(cms.PSet(
        PlotBounds = cms.vdouble(0.0, 0.0),
        HLTCollectionLabels = cms.InputTag("hltL1sSingleEgamma","","myHLT"),
        IsoCollections = cms.VInputTag(cms.InputTag("none")),
        theHLTOutputTypes = cms.uint32(83)
    ), 
        cms.PSet(
            PlotBounds = cms.vdouble(0.0, 0.0),
            HLTCollectionLabels = cms.InputTag("hltL1IsoLargeWindowSingleL1MatchFilter","","myHLT"),
            IsoCollections = cms.VInputTag(cms.InputTag("none")),
            theHLTOutputTypes = cms.uint32(100)
        ), 
        cms.PSet(
            PlotBounds = cms.vdouble(0.0, 0.0),
            HLTCollectionLabels = cms.InputTag("hltL1IsoLargeWindowSingleElectronEtFilter","","myHLT"),
            IsoCollections = cms.VInputTag(cms.InputTag("none")),
            theHLTOutputTypes = cms.uint32(100)
        ), 
        cms.PSet(
            PlotBounds = cms.vdouble(0.0, 10.0),
            HLTCollectionLabels = cms.InputTag("hltL1IsoLargeWindowSingleElectronHcalIsolFilter","","myHLT"),
            IsoCollections = cms.VInputTag(cms.InputTag("l1IsolatedElectronHcalIsol","","myHLT")),
            theHLTOutputTypes = cms.uint32(100)
        ), 
        cms.PSet(
            PlotBounds = cms.vdouble(0.0, 0.0),
            HLTCollectionLabels = cms.InputTag("hltL1IsoLargeWindowSingleElectronPixelMatchFilter","","myHLT"),
            IsoCollections = cms.VInputTag(cms.InputTag("none")),
            theHLTOutputTypes = cms.uint32(100)
        ), 
        cms.PSet(
            PlotBounds = cms.vdouble(0.0, 0.0),
            HLTCollectionLabels = cms.InputTag("hltL1IsoLargeWindowSingleElectronHOneOEMinusOneOPFilter","","myHLT"),
            IsoCollections = cms.VInputTag(cms.InputTag("none")),
            theHLTOutputTypes = cms.uint32(92)
        ), 
        cms.PSet(
            PlotBounds = cms.vdouble(0.0, 0.0),
            HLTCollectionLabels = cms.InputTag("hltL1IsoLargeWindowSingleElectronDetaDphiFilter","","myHLT"),
            IsoCollections = cms.VInputTag(cms.InputTag("none")),
            theHLTOutputTypes = cms.uint32(92)
        ), 
        cms.PSet(
            PlotBounds = cms.vdouble(0.0, 10.0),
            HLTCollectionLabels = cms.InputTag("hltL1IsoLargeWindowSingleElectronTrackIsolFilter","","myHLT"),
            IsoCollections = cms.VInputTag(cms.InputTag("l1IsoLargeWindowElectronTrackIsol","","myHLT")),
            theHLTOutputTypes = cms.uint32(92)
        )),
    PtMax = cms.untracked.double(200.0),
    pdgGen = cms.int32(11)
)


