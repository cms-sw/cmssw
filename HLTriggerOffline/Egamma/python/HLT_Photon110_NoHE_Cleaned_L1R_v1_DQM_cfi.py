#----------------------------------------
# path HLT_Photon110_NoHE_Cleaned_L1R_v1
#----------------------------------------

import FWCore.ParameterSet.Config as cms

HLT_Photon110_NoHE_Cleaned_L1R_v1_DQM = cms.EDAnalyzer("EmDQM",
    genEtaAcc = cms.double(2.5),
    triggerobject = cms.InputTag("hltTriggerSummaryRAW","","HLT"),
    cutnum = cms.int32(1),
    genEtMin = cms.untracked.double(110.0),
    cutcollection = cms.InputTag("fiducialGammaJet"),
    genEtAcc = cms.double(2.0),
    reqNum = cms.uint32(1),
    filters = cms.VPSet(cms.PSet(
        PlotBounds = cms.vdouble(0.0, 0.0),
        theHLTOutputTypes = cms.int32(-82),
        IsoCollections = cms.VInputTag(cms.InputTag("none")),
        HLTCollectionLabels = cms.InputTag("hltL1sL1SingleEG8","","HLT")
    ), 
        cms.PSet(
            PlotBounds = cms.vdouble(0.0, 0.0),
            theHLTOutputTypes = cms.int32(92),
            IsoCollections = cms.VInputTag(cms.InputTag("none")),
            HLTCollectionLabels = cms.InputTag("hltL1NonIsoHLTNonIsoSinglePhotonEt110NoHECleanedL1MatchFilterRegional","","HLT")
        ), 
        cms.PSet(
            PlotBounds = cms.vdouble(0.0, 0.0),
            theHLTOutputTypes = cms.int32(92),
            IsoCollections = cms.VInputTag(cms.InputTag("none")),
            HLTCollectionLabels = cms.InputTag("hltL1NonIsoHLTNonIsoSinglePhotonEt110NoHECleanedEtFilter","","HLT")
        ), 
        cms.PSet(
            PlotBounds = cms.vdouble(0.0, 0.0),
            theHLTOutputTypes = cms.int32(92),
            IsoCollections = cms.VInputTag(cms.InputTag("hltL1IsoR9shape"), cms.InputTag("hltL1NonIsoR9shape")),
            HLTCollectionLabels = cms.InputTag("hltL1NonIsoHLTNonIsoSinglePhotonEt110NoHECleanedR9ShapeFilter","","HLT")
        ), 
        cms.PSet(
            PlotBounds = cms.vdouble(0.0, 0.0),
            theHLTOutputTypes = cms.int32(81),
            IsoCollections = cms.VInputTag(cms.InputTag("hltL1IsolatedPhotonHcalForHE"), cms.InputTag("hltL1NonIsolatedPhotonHcalForHE")),
            HLTCollectionLabels = cms.InputTag("hltL1NonIsoHLTNonIsoSinglePhotonEt110NoHECleanedHEFilter","","HLT")
        )),
    PtMax = cms.untracked.double(100.0),
    pdgGen = cms.int32(22)
)
