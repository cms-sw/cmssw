import FWCore.ParameterSet.Config as cms

# RecoMuon flux ##########################################################
# Common HLT stuff
from HLTrigger.Configuration.common.CaloTowers_cff import *
# L2 reconstruction
from RecoMuon.L2MuonProducer.L2Muons_cff import *
import copy
from RecoMuon.L2MuonProducer.L2Muons_cfi import *
hltL2Muons = copy.deepcopy(L2Muons)
# L2 candidates
hltL2MuonCandidates = cms.EDProducer("L2MuonCandidateProducer",
    InputObjects = cms.InputTag("hltL2Muons","UpdatedAtVtx")
)

# L2 calorimeter isolation
hltL2MuonIsolations = cms.EDProducer("L2MuonIsolationProducer",
    ConeSizes = cms.vdouble(0.24, 0.24, 0.24, 0.24, 0.24, 
        0.24, 0.24, 0.24, 0.24, 0.24, 
        0.24, 0.24, 0.24, 0.24, 0.24, 
        0.24, 0.24, 0.24, 0.24, 0.24, 
        0.24, 0.24, 0.24, 0.24, 0.24, 
        0.24),
    Thresholds = cms.vdouble(4.0, 3.7, 4.0, 3.5, 3.4, 
        3.4, 3.2, 3.4, 3.1, 2.9, 
        2.9, 2.7, 3.1, 3.0, 2.4, 
        2.1, 2.0, 2.3, 2.2, 2.4, 
        2.5, 2.5, 2.6, 2.9, 3.1, 
        2.9),
    StandAloneCollectionLabel = cms.InputTag("hltL2Muons","UpdatedAtVtx"),
    OutputMuIsoDeposits = cms.bool(True),
    EtaBounds = cms.vdouble(0.0435, 0.1305, 0.2175, 0.3045, 0.3915, 
        0.4785, 0.5655, 0.6525, 0.7395, 0.8265, 
        0.9135, 1.0005, 1.0875, 1.1745, 1.2615, 
        1.3485, 1.4355, 1.5225, 1.6095, 1.6965, 
        1.785, 1.88, 1.9865, 2.1075, 2.247, 
        2.411),
    ExtractorPSet = cms.PSet(
        DR_Veto_H = cms.double(0.1),
        Vertex_Constraint_Z = cms.bool(False),
        Threshold_H = cms.double(0.5),
        ComponentName = cms.string('CaloExtractor'),
        Threshold_E = cms.double(0.2),
        DR_Max = cms.double(0.24),
        DR_Veto_E = cms.double(0.07),
        Weight_E = cms.double(1.5),
        Vertex_Constraint_XY = cms.bool(False),
        DepositLabel = cms.untracked.string('EcalPlusHcal'),
        CaloTowerCollectionLabel = cms.InputTag("towerMakerForMuons"),
        Weight_H = cms.double(1.0)
    )
)

# L3 candidates
hltL3MuonCandidates = cms.EDProducer("L3MuonCandidateProducer",
    InputObjects = cms.InputTag("hltL3Muons")
)

# L3 track isolation
hltL3MuonIsolations = cms.EDProducer("L3MuonIsolationProducer",
    inputMuonCollection = cms.InputTag("hltL3Muons"),
    CutsPSet = cms.PSet(
        ConeSizes = cms.vdouble(0.24, 0.24, 0.24, 0.24, 0.24, 
            0.24, 0.24, 0.24, 0.24, 0.24, 
            0.24, 0.24, 0.24, 0.24, 0.24, 
            0.24, 0.24, 0.24, 0.24, 0.24, 
            0.24, 0.24, 0.24, 0.24, 0.24, 
            0.24),
        ComponentName = cms.string('SimpleCuts'),
        EtaBounds = cms.vdouble(0.0435, 0.1305, 0.2175, 0.3045, 0.3915, 
            0.4785, 0.5655, 0.6525, 0.7395, 0.8265, 
            0.9135, 1.0005, 1.0875, 1.1745, 1.2615, 
            1.3485, 1.4355, 1.5225, 1.6095, 1.6965, 
            1.785, 1.88, 1.9865, 2.1075, 2.247, 
            2.411),
        Thresholds = cms.vdouble(1.1, 1.1, 1.1, 1.1, 1.2, 
            1.1, 1.2, 1.1, 1.2, 1.0, 
            1.1, 1.0, 1.0, 1.1, 1.0, 
            1.0, 1.1, 0.9, 1.1, 0.9, 
            1.1, 1.0, 1.0, 0.9, 0.8, 
            0.1)
    ),
    TrackPt_Min = cms.double(-1.0),
    OutputMuIsoDeposits = cms.bool(True),
    ExtractorPSet = cms.PSet(
        Diff_z = cms.double(0.2),
        inputTrackCollection = cms.InputTag("pixelTracks"),
        BeamSpotLabel = cms.InputTag("offlineBeamSpot"),
        ComponentName = cms.string('TrackExtractor'),
        DR_Max = cms.double(0.24),
        #double Pt_Min = 0.9
        Diff_r = cms.double(0.1),
        Chi2Prob_Min = cms.double(-1.0),
        DR_Veto = cms.double(0.01),
        NHits_Min = cms.uint32(0),
        Chi2Ndof_Max = cms.double(1e+64),
        Pt_Min = cms.double(-1.0),
        DepositLabel = cms.untracked.string('PXLS'),
        BeamlineOption = cms.string('BeamSpotFromEvent')
    )
)

l1muonreco = cms.Sequence(cms.SequencePlaceholder("hltBegin"))
l2muonrecoNocand = cms.Sequence(cms.SequencePlaceholder("doLocalMuon")*cms.SequencePlaceholder("hltL2MuonSeeds")*hltL2Muons)
l2muonreco = cms.Sequence(l2muonrecoNocand*hltL2MuonCandidates)
l2muonisoreco = cms.Sequence(doRegionalCaloForMuons*hltL2MuonIsolations)
l3muonrecoNocand = cms.Sequence(cms.SequencePlaceholder("doLocalPixel")+cms.SequencePlaceholder("doLocalStrip")+cms.SequencePlaceholder("hltL3MuonTracks"))
l3muonreco = cms.Sequence(l3muonrecoNocand*hltL3MuonCandidates)
l3muonisoreco = cms.Sequence(cms.SequencePlaceholder("pixelTracksForMuons")*hltL3MuonIsolations)
hltL2Muons.InputObjects = 'hltL2MuonSeeds'

