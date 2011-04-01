#----------------------------------------
# path HLT_DoubleEle8_SW_HT70U_L1R_v1
#----------------------------------------

import FWCore.ParameterSet.Config as cms

HLT_DoubleEle8_SW_HT70U_L1R_v1_DQM = cms.EDAnalyzer("EmDQM",
    genEtaAcc = cms.double(2.5),
    triggerobject = cms.InputTag("hltTriggerSummaryRAW","","HLT"),
    cutnum = cms.int32(2),
    genEtMin = cms.untracked.double(8.0),
    cutcollection = cms.InputTag("fiducialZee"),
    genEtAcc = cms.double(2.0),
    reqNum = cms.uint32(2),
    filters = cms.VPSet(cms.PSet(
        PlotBounds = cms.vdouble(0.0, 0.0),
        theHLTOutputTypes = cms.int32(-82),
        IsoCollections = cms.VInputTag(cms.InputTag("none")),
        HLTCollectionLabels = cms.InputTag("hltL1sL1DoubleEG5","","HLT")
    ), 
        cms.PSet(
            PlotBounds = cms.vdouble(0.0, 0.0),
            theHLTOutputTypes = cms.int32(92),
            IsoCollections = cms.VInputTag(cms.InputTag("none")),
            HLTCollectionLabels = cms.InputTag("hltL1NonIsoHLTNonIsoDoubleElectronEt8HT70L1MatchFilterRegional","","HLT")
        ), 
        cms.PSet(
            PlotBounds = cms.vdouble(0.0, 0.0),
            theHLTOutputTypes = cms.int32(92),
            IsoCollections = cms.VInputTag(cms.InputTag("none")),
            HLTCollectionLabels = cms.InputTag("hltL1NonIsoHLTNonIsoDoubleElectronEt8HT70EtFilter","","HLT")
        ), 
        cms.PSet(
            PlotBounds = cms.vdouble(0.0, 0.0),
            theHLTOutputTypes = cms.int32(92),
            IsoCollections = cms.VInputTag(cms.InputTag("hltL1IsolatedPhotonHcalForHE"), cms.InputTag("hltL1NonIsolatedPhotonHcalForHE")),
            HLTCollectionLabels = cms.InputTag("hltL1NonIsoHLTNonIsoDoubleElectronEt8HT70HEFilter","","HLT")
        ), 
        cms.PSet(
            PlotBounds = cms.vdouble(0.0, 0.0),
            theHLTOutputTypes = cms.int32(92),
            IsoCollections = cms.VInputTag(cms.InputTag("none")),
            HLTCollectionLabels = cms.InputTag("hltL1NonIsoHLTNonIsoDoubleElectronEt8HT70PixelMatchFilter","","HLT")
        )),
    PtMax = cms.untracked.double(100.0),
    pdgGen = cms.int32(11)
)
