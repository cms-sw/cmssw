import FWCore.ParameterSet.Config as cms

hltMonMu = cms.EDAnalyzer("HLTMon",
    outputFile = cms.untracked.string('./L1TDQM.root'),
    verbose = cms.untracked.bool(False),
    MonitorDaemon = cms.untracked.bool(True),
    reqNum = cms.uint32(1),
    DaqMonitorBEInterface = cms.untracked.bool(True),
    filters = cms.VPSet(cms.PSet(
        PlotBounds = cms.vdouble(0.0, 0.0),
        HLTCollectionLabels = cms.InputTag("hltZMML2Filtered","","HLT"),
        IsoCollections = cms.VInputTag(cms.InputTag("none")),
        theHLTOutputTypes = cms.uint32(93)
    ), 
        cms.PSet(
            PlotBounds = cms.vdouble(0.0, 0.0),
            HLTCollectionLabels = cms.InputTag("hltEMuL1MuonFilter","","HLT"),
            IsoCollections = cms.VInputTag(cms.InputTag("none")),
            theHLTOutputTypes = cms.uint32(81)
        ), 
        cms.PSet(
            PlotBounds = cms.vdouble(0.0, 0.0),
            HLTCollectionLabels = cms.InputTag("hltSingleMuNoIsoL1Filtered","","HLT"),
            IsoCollections = cms.VInputTag(cms.InputTag("hltSingleMuNoIsoLevel1Seed")),
            theHLTOutputTypes = cms.uint32(81)
        ), 
        cms.PSet(
            PlotBounds = cms.vdouble(0.0, 0.0),
            HLTCollectionLabels = cms.InputTag("hltSingleMuIsoL2IsoFiltered","","HLT"),
            IsoCollections = cms.VInputTag(cms.InputTag("none")),
            theHLTOutputTypes = cms.uint32(93)
        ), 
        cms.PSet(
            PlotBounds = cms.vdouble(0.0, 0.0),
            HLTCollectionLabels = cms.InputTag("hltSingleMuIsoL3IsoFiltered","","HLT"),
            IsoCollections = cms.VInputTag(cms.InputTag("none")),
            theHLTOutputTypes = cms.uint32(93)
        ), 
        cms.PSet(
            PlotBounds = cms.vdouble(0.0, 0.0),
            HLTCollectionLabels = cms.InputTag("hltSingleMuNoIsoL1Filtered10","","HLT"),
            IsoCollections = cms.VInputTag(cms.InputTag("hltSingleMuNoIsoLevel1Seed10")),
            theHLTOutputTypes = cms.uint32(81)
        ), 
        cms.PSet(
            PlotBounds = cms.vdouble(0.0, 0.0),
            HLTCollectionLabels = cms.InputTag("hltSingleMuIsoL2IsoFiltered13","","HLT"),
            IsoCollections = cms.VInputTag(cms.InputTag("none")),
            theHLTOutputTypes = cms.uint32(93)
        ), 
        cms.PSet(
            PlotBounds = cms.vdouble(0.0, 0.0),
            HLTCollectionLabels = cms.InputTag("hltSingleMuIsoL3IsoFiltered13","","HLT"),
            IsoCollections = cms.VInputTag(cms.InputTag("none")),
            theHLTOutputTypes = cms.uint32(93)
        ), 
        cms.PSet(
            PlotBounds = cms.vdouble(0.0, 0.0),
            HLTCollectionLabels = cms.InputTag("hltSingleMuIsoL2IsoFiltered15","","HLT"),
            IsoCollections = cms.VInputTag(cms.InputTag("none")),
            theHLTOutputTypes = cms.uint32(93)
        ), 
        cms.PSet(
            PlotBounds = cms.vdouble(0.0, 0.0),
            HLTCollectionLabels = cms.InputTag("hltSingleMuIsoL3IsoFiltered15","","HLT"),
            IsoCollections = cms.VInputTag(cms.InputTag("none")),
            theHLTOutputTypes = cms.uint32(93)
        ), 
        cms.PSet(
            PlotBounds = cms.vdouble(0.0, 0.0),
            HLTCollectionLabels = cms.InputTag("hltSingleMuIsoL2IsoFiltered9","","HLT"),
            IsoCollections = cms.VInputTag(cms.InputTag("none")),
            theHLTOutputTypes = cms.uint32(93)
        ), 
        cms.PSet(
            PlotBounds = cms.vdouble(0.0, 0.0),
            HLTCollectionLabels = cms.InputTag("hltSingleMuIsoL3OIstateIsoFiltered","","HLT"),
            IsoCollections = cms.VInputTag(cms.InputTag("none")),
            theHLTOutputTypes = cms.uint32(93)
        ), 
        cms.PSet(
            PlotBounds = cms.vdouble(0.0, 0.0),
            HLTCollectionLabels = cms.InputTag("hltMuLevel1PathL1OpenFiltered","","HLT"),
            IsoCollections = cms.VInputTag(cms.InputTag("hltMuLevel1PathLevel1OpenSeed")),
            theHLTOutputTypes = cms.uint32(81)
        ), 
        cms.PSet(
            PlotBounds = cms.vdouble(0.0, 0.0),
            HLTCollectionLabels = cms.InputTag("hltMuLevel1PathL1Filtered","","HLT"),
            IsoCollections = cms.VInputTag(cms.InputTag("hltMuLevel1PathLevel1Seed")),
            theHLTOutputTypes = cms.uint32(81)
        ), 
        cms.PSet(
            PlotBounds = cms.vdouble(0.0, 0.0),
            HLTCollectionLabels = cms.InputTag("hltSingleMuPrescale5L1Filtered","","HLT"),
            IsoCollections = cms.VInputTag(cms.InputTag("hltSingleMuPrescale5Level1Seed")),
            theHLTOutputTypes = cms.uint32(81)
        ), 
        cms.PSet(
            PlotBounds = cms.vdouble(0.0, 0.0),
            HLTCollectionLabels = cms.InputTag("hltSingleMuPrescale710L1Filtered","","HLT"),
            IsoCollections = cms.VInputTag(cms.InputTag("hltSingleMuPrescale710Level1Seed")),
            theHLTOutputTypes = cms.uint32(81)
        ), 
        cms.PSet(
            PlotBounds = cms.vdouble(0.0, 0.0),
            HLTCollectionLabels = cms.InputTag("hltSingleMuPrescale77L1Filtered","","HLT"),
            IsoCollections = cms.VInputTag(cms.InputTag("hltSingleMuPrescale77Level1Seed")),
            theHLTOutputTypes = cms.uint32(81)
        )),
    disableROOToutput = cms.untracked.bool(True)
)


