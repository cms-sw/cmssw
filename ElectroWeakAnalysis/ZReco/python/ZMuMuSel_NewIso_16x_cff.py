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
zMuMuIsolations.CutsPSet.EtaBounds = [999.9]
zMuMuIsolations.CutsPSet.ConeSizes = [0.3]
zMuMuIsolations.CutsPSet.Thresholds = [3.0]
zMuMuIsolations.OutputMuIsoDeposits = False
zMuMuIsolations.ExtractorPSet.inputTrackCollection = 'ctfWithMaterialTracks'
zMuMuTrackerIsolations.inputMuonCollection = 'ctfWithMaterialTracks'
zMuMuTrackerIsolations.CutsPSet.EtaBounds = [999.9]
zMuMuTrackerIsolations.CutsPSet.ConeSizes = [0.3]
zMuMuTrackerIsolations.CutsPSet.Thresholds = [3.0]
zMuMuTrackerIsolations.OutputMuIsoDeposits = False
zMuMuTrackerIsolations.ExtractorPSet.inputTrackCollection = 'ctfWithMaterialTracks'

