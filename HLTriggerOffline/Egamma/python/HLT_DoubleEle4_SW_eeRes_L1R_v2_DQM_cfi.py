#----------------------------------------
# path HLT_DoubleEle4_SW_eeRes_L1R_v2
#----------------------------------------

import FWCore.ParameterSet.Config as cms

HLT_DoubleEle4_SW_eeRes_L1R_v2_DQM = cms.EDAnalyzer("EmDQM",
    genEtaAcc = cms.double(2.5),
    triggerobject = cms.InputTag("hltTriggerSummaryRAW","","HLT"),
    cutnum = cms.int32(2),
    genEtMin = cms.untracked.double(4.0),
    cutcollection = cms.InputTag("fiducialZee"),
    genEtAcc = cms.double(2.0),
    reqNum = cms.uint32(2),
    filters = cms.VPSet(cms.PSet(
        PlotBounds = cms.vdouble(0.0, 0.0),
        theHLTOutputTypes = cms.int32(-82),
        IsoCollections = cms.VInputTag(cms.InputTag("none")),
        HLTCollectionLabels = cms.InputTag("hltL1sL1DoubleEG2","","HLT")
    ), 
        cms.PSet(
            PlotBounds = cms.vdouble(0.0, 0.0),
            theHLTOutputTypes = cms.int32(92),
            IsoCollections = cms.VInputTag(cms.InputTag("none")),
            HLTCollectionLabels = cms.InputTag("hltL1NonIsoDoublePhotonEt4eeResL1MatchFilterRegional","","HLT")
        ), 
        cms.PSet(
            PlotBounds = cms.vdouble(0.0, 0.0),
            theHLTOutputTypes = cms.int32(92),
            IsoCollections = cms.VInputTag(cms.InputTag("none")),
            HLTCollectionLabels = cms.InputTag("hltL1NonIsoDoublePhotonEt4eeResEtFilter","","HLT")
        ), 
        cms.PSet(
            PlotBounds = cms.vdouble(0.0, 0.0),
            theHLTOutputTypes = cms.int32(92),
            IsoCollections = cms.VInputTag(cms.InputTag("hltL1IsoR9shapeLowPt"), cms.InputTag("hltL1NonIsoR9shapeLowPt")),
            HLTCollectionLabels = cms.InputTag("hltL1NonIsoHLTNonIsoDoublePhotonEt4eeResR9ShapeFilter","","HLT")
        ), 
        cms.PSet(
            PlotBounds = cms.vdouble(0.0, 0.0),
            theHLTOutputTypes = cms.int32(92),
            IsoCollections = cms.VInputTag(cms.InputTag("hltL1IsoHLTClusterShapeLowPt"), cms.InputTag("hltL1NonIsoHLTClusterShapeLowPt")),
            HLTCollectionLabels = cms.InputTag("hltL1NonIsoDoublePhotonEt4eeResClusterShapeFilter","","HLT")
        ), 
        cms.PSet(
            PlotBounds = cms.vdouble(0.0, 0.0),
            theHLTOutputTypes = cms.int32(92),
            IsoCollections = cms.VInputTag(cms.InputTag("hltL1IsolatedPhotonEcalIsolLowPt"), cms.InputTag("hltL1NonIsolatedPhotonEcalIsolLowPt")),
            HLTCollectionLabels = cms.InputTag("hltL1NonIsoDoublePhotonEt4eeResEcalIsolFilter","","HLT")
        ), 
        cms.PSet(
            PlotBounds = cms.vdouble(0.0, 0.0),
            theHLTOutputTypes = cms.int32(92),
            IsoCollections = cms.VInputTag(cms.InputTag("hltL1IsolatedElectronHcalIsolLowPt"), cms.InputTag("hltL1NonIsolatedElectronHcalIsolLowPt")),
            HLTCollectionLabels = cms.InputTag("hltL1NonIsoDoublePhotonEt4eeResHcalIsolFilter","","HLT")
        ), 
        cms.PSet(
            PlotBounds = cms.vdouble(0.0, 0.0),
            theHLTOutputTypes = cms.int32(92),
            IsoCollections = cms.VInputTag(cms.InputTag("none")),
            HLTCollectionLabels = cms.InputTag("hltL1NonIsoDoublePhotonEt4eeResPixelMatchFilter","","HLT")
        ), 
        cms.PSet(
            PlotBounds = cms.vdouble(0.0, 0.0),
            theHLTOutputTypes = cms.int32(82),
            IsoCollections = cms.VInputTag(cms.InputTag("none")),
            HLTCollectionLabels = cms.InputTag("hltL1NonIsoDoublePhotonEt4eeResOneOEMinusOneOPFilter","","HLT")
        )),
    PtMax = cms.untracked.double(100.0),
    pdgGen = cms.int32(11)
)
