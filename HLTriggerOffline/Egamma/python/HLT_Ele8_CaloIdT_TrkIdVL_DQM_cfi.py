import FWCore.ParameterSet.Config as cms

HLT_Ele8_CaloIdT_TrkIdVL_DQM = cms.EDAnalyzer("EmDQM",
    triggerobject = cms.InputTag("hltTriggerSummaryRAW","","HLT"),                            
    genEtaAcc = cms.double(2.5),
    genEtAcc = cms.double(2.0),
    genEtMin = cms.untracked.double(8.0),
    reqNum = cms.uint32(1),
    cutcollection = cms.InputTag("fiducialWenu","",""),
    cutnum = cms.int32(1),
    filters = cms.VPSet(cms.PSet(
        PlotBounds = cms.vdouble(0.0, 0.0),
        HLTCollectionLabels = cms.InputTag("hltL1sL1SingleEG5","","HLT"),
        IsoCollections = cms.VInputTag(cms.InputTag("none")),
        theHLTOutputTypes = cms.int32(-82),
        ncandcut = cms.int32(1)
    ), 
        cms.PSet(
            PlotBounds = cms.vdouble(0.0, 0.0),
            HLTCollectionLabels = cms.InputTag("hltEGRegionalL1SingleEG5","","HLT"),
            IsoCollections = cms.VInputTag(cms.InputTag("none")),
            theHLTOutputTypes = cms.int32(92),
            ncandcut = cms.int32(1)
        ), 
        cms.PSet(
            PlotBounds = cms.vdouble(0.0, 0.0),
            HLTCollectionLabels = cms.InputTag("hltEG8EtFilter","","HLT"),
            IsoCollections = cms.VInputTag(cms.InputTag("none")),
            theHLTOutputTypes = cms.int32(92),
            ncandcut = cms.int32(1)
        ), 
        cms.PSet(
            PlotBounds = cms.vdouble(0.0, 0.0),
            HLTCollectionLabels = cms.InputTag("hltEle8CaloIdTTrkIdVLClusterShapeFilter","","HLT"),
            IsoCollections = cms.VInputTag(cms.InputTag("hltL1SeededHLTClusterShape","","HLT")),
            theHLTOutputTypes = cms.int32(92),
            ncandcut = cms.int32(1)
        ), 
        cms.PSet(
            PlotBounds = cms.vdouble(0.0, 0.0),
            HLTCollectionLabels = cms.InputTag("hltEle8CaloIdTTrkIdVLDphiFilter","","HLT"),
            IsoCollections = cms.VInputTag(cms.InputTag("hltElectronL1SeededDetaDphi:Dphi")),
            theHLTOutputTypes = cms.int32(82),
            ncandcut = cms.int32(1)
        )),
    PtMax = cms.untracked.double(100.0),
    pdgGen = cms.int32(11)
)



