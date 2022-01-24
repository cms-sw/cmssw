import FWCore.ParameterSet.Config as cms
from DQMServices.Core.DQMEDAnalyzer import DQMEDAnalyzer
from RecoMuon.TrackingTools.MuonServiceProxy_cff import MuonServiceProxy
from DQM.GEM.gemEfficiencyAnalyzerDefault_cfi import gemEfficiencyAnalyzerDefault as _gemEfficiencyAnalyzerDefault
from TrackingTools.TransientTrack.TransientTrackBuilder_cfi import *

gemOfflineDQMTightGlbMuons = cms.EDFilter("MuonSelector",
    src = cms.InputTag('muons'),
    cut = cms.string(
        '(pt > 20)'
        '&& isGlobalMuon'
        '&& globalTrack.isNonnull'
        '&& passed(\'CutBasedIdTight\')'
    ),
    filter = cms.bool(False)
)

gemOfflineDQMStaMuons = cms.EDFilter("MuonSelector",
    src = cms.InputTag('muons'),
    cut = cms.string(
        '(pt > 20)'
        '&& isStandAloneMuon'
        '&& outerTrack.isNonnull'
    ),
    filter = cms.bool(False)
)

gemEfficiencyAnalyzerTightGlb = _gemEfficiencyAnalyzerDefault.clone(
    ServiceParameters = MuonServiceProxy.ServiceParameters.clone(),
    folder = 'GEM/Efficiency/type1',
    muonTag = 'gemOfflineDQMTightGlbMuons',
    name = 'Tight GLB Muon',
    useGlobalMuon = True
)

gemEfficiencyAnalyzerSta = _gemEfficiencyAnalyzerDefault.clone(
    ServiceParameters = MuonServiceProxy.ServiceParameters.clone(),
    muonTag = "gemOfflineDQMStaMuons",
    folder = 'GEM/Efficiency/type2',
    name = 'STA Muon',
    useGlobalMuon = False
)

from Configuration.Eras.Modifier_phase2_GEM_cff import phase2_GEM
phase2_GEM.toModify(
    gemEfficiencyAnalyzerTightGlb,
    etaNbins=cms.untracked.int32(15),
    etaUp=cms.untracked.double(3.0))

phase2_GEM.toModify(
    gemEfficiencyAnalyzerSta,
    etaNbins=cms.untracked.int32(15),
    etaUp=cms.untracked.double(3.0))

gemEfficiencyAnalyzerTightGlbSeq = cms.Sequence(
    cms.ignore(gemOfflineDQMTightGlbMuons) *
    gemEfficiencyAnalyzerTightGlb)

gemEfficiencyAnalyzerStaSeq = cms.Sequence(
    cms.ignore(gemOfflineDQMStaMuons) *
    gemEfficiencyAnalyzerSta)
