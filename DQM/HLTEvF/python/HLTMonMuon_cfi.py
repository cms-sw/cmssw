import FWCore.ParameterSet.Config as cms

hltMonMu = cms.EDAnalyzer("HLTMon",
    outputFile = cms.untracked.string('./L1TDQM.root'),
    verbose = cms.untracked.bool(False),
    MonitorDaemon = cms.untracked.bool(True),
    reqNum = cms.uint32(1),
    DaqMonitorBEInterface = cms.untracked.bool(True),
    filters = cms.VPSet(
#muon open
        cms.PSet(
            PlotBounds = cms.vdouble(0.0, 0.0),
            HLTCollectionLabels = cms.InputTag("hltMuLevel1PathL1OpenFiltered","","HLT"),
            IsoCollections = cms.VInputTag(cms.InputTag("none")),
            theHLTOutputTypes = cms.uint32(81)
        ), 
#HLT_L2Mu9
        cms.PSet(
            PlotBounds = cms.vdouble(0.0, 0.0),
            HLTCollectionLabels = cms.InputTag("hltSingleMuNoIsoL1Filtered","","HLT"),
            IsoCollections = cms.VInputTag(cms.InputTag("none")),
            theHLTOutputTypes = cms.uint32(81)
        ), 
        cms.PSet(
            PlotBounds = cms.vdouble(0.0, 0.0),
            HLTCollectionLabels = cms.InputTag("hltSingleMuLevel2NoIsoL2PreFiltered","","HLT"),
            IsoCollections = cms.VInputTag(cms.InputTag("none")),
            theHLTOutputTypes = cms.uint32(93)
        ), 
#HLT_Mu3	
        cms.PSet(
            PlotBounds = cms.vdouble(0.0, 0.0),
            HLTCollectionLabels = cms.InputTag("hltSingleMuIsoL3IsoFiltered","","HLT"),
            IsoCollections = cms.VInputTag(cms.InputTag("none")),
            theHLTOutputTypes = cms.uint32(93)
        ), 
        cms.PSet(
            PlotBounds = cms.vdouble(0.0, 0.0),
            HLTCollectionLabels = cms.InputTag("hltSingleMuPrescale3L2PreFiltered","","HLT"),
            IsoCollections = cms.VInputTag(cms.InputTag("none")),
            theHLTOutputTypes = cms.uint32(93)
        )),
    disableROOToutput = cms.untracked.bool(True)
)


