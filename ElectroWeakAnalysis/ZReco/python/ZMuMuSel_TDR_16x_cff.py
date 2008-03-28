import FWCore.ParameterSet.Config as cms

import copy
from RecoMuon.L3MuonIsolationProducer.L3MuonIsolationProducer_cfi import *
# TDR selection procedure for Z->MuMu events
# Tested on 2007/10/08/ by J.A.
# Track isolation
zMuMuIsolations = copy.deepcopy(L3MuonIsolations)
import copy
from RecoMuon.L3MuonIsolationProducer.L3MuonIsolationProducer_cfi import *
zMuMuTrackerIsolations = copy.deepcopy(L3MuonIsolations)
zMuMuSelFilter = cms.EDFilter("ZToMuMuSelector",
    MassZMin = cms.double(83.7),
    TrackerTag = cms.untracked.InputTag("ctfWithMaterialTracks"),
    MinTrackerHits = cms.untracked.int32(7),
    MuonTag = cms.InputTag("globalMuons"),
    OnlyGlobalMuons = cms.bool(False),
    EtaCut = cms.double(2.0),
    MassZMax = cms.double(98.7),
    PtCut = cms.double(20.0),
    IsolationTag = cms.InputTag("zMuMuIsolations"),
    TrackerIsolationTag = cms.untracked.InputTag("zMuMuTrackerIsolations")
)

zMuMuSelGlobalOnlyFilter = cms.EDFilter("ZToMuMuSelector",
    EtaCut = cms.double(2.0),
    MuonTag = cms.InputTag("globalMuons"),
    OnlyGlobalMuons = cms.bool(True),
    MassZMin = cms.double(83.7),
    MassZMax = cms.double(98.7),
    PtCut = cms.double(20.0),
    IsolationTag = cms.InputTag("zMuMuIsolations")
)

zmumuselGlobalOnly = cms.Sequence(zMuMuIsolations*zMuMuSelGlobalOnlyFilter)
zmumusel = cms.Sequence(zMuMuIsolations*zMuMuTrackerIsolations*zMuMuSelFilter)
zMuMuIsolations.inputMuonCollection = 'globalMuons'
zMuMuIsolations.CutsPSet.EtaBounds = [0.0435, 0.1305, 0.2175, 0.3045, 0.3915, 0.4785, 0.5655, 0.6525, 0.7395, 0.8265, 0.9135, 1.0005, 1.0875, 1.1745, 1.2615, 1.3485, 1.4355, 1.5225, 1.6095, 1.6965, 1.7850, 1.8800, 1.9865, 2.1075, 2.2470, 2.4110]
zMuMuIsolations.CutsPSet.ConeSizes = [0.24, 0.24, 0.24, 0.24, 0.24, 0.24, 0.24, 0.24, 0.24, 0.24, 0.24, 0.24, 0.24, 0.24, 0.24, 0.24, 0.24, 0.24, 0.24, 0.24, 0.24, 0.24, 0.24, 0.24, 0.24, 0.24]
zMuMuIsolations.CutsPSet.Thresholds = [3.20, 3.10, 2.80, 3.00, 3.30, 3.30, 2.80, 3.30, 2.90, 2.20, 2.50, 2.70, 2.90, 2.70, 2.50, 2.80, 2.30, 2.60, 2.70, 2.50, 2.40, 2.90, 2.50, 2.70, 2.10, 1.60]
zMuMuIsolations.OutputMuIsoDeposits = False
zMuMuIsolations.ExtractorPSet.inputTrackCollection = 'ctfWithMaterialTracks'
zMuMuTrackerIsolations.inputMuonCollection = 'ctfWithMaterialTracks'
zMuMuTrackerIsolations.CutsPSet.EtaBounds = [0.0435, 0.1305, 0.2175, 0.3045, 0.3915, 0.4785, 0.5655, 0.6525, 0.7395, 0.8265, 0.9135, 1.0005, 1.0875, 1.1745, 1.2615, 1.3485, 1.4355, 1.5225, 1.6095, 1.6965, 1.7850, 1.8800, 1.9865, 2.1075, 2.2470, 2.4110]
zMuMuTrackerIsolations.CutsPSet.ConeSizes = [0.24, 0.24, 0.24, 0.24, 0.24, 0.24, 0.24, 0.24, 0.24, 0.24, 0.24, 0.24, 0.24, 0.24, 0.24, 0.24, 0.24, 0.24, 0.24, 0.24, 0.24, 0.24, 0.24, 0.24, 0.24, 0.24]
zMuMuTrackerIsolations.CutsPSet.Thresholds = [3.20, 3.10, 2.80, 3.00, 3.30, 3.30, 2.80, 3.30, 2.90, 2.20, 2.50, 2.70, 2.90, 2.70, 2.50, 2.80, 2.30, 2.60, 2.70, 2.50, 2.40, 2.90, 2.50, 2.70, 2.10, 1.60]
zMuMuTrackerIsolations.OutputMuIsoDeposits = False
zMuMuTrackerIsolations.ExtractorPSet.inputTrackCollection = 'ctfWithMaterialTracks'

