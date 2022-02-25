import FWCore.ParameterSet.Config as cms
from DQM.GEM.gemEfficiencyAnalyzer_cfi import *

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

gemEfficiencyAnalyzerTightGlb = gemEfficiencyAnalyzer.clone(
    folder = 'GEM/Efficiency/type1',
    muonTag = 'gemOfflineDQMTightGlbMuons',
    name = 'Tight GLB Muon',
    useGlobalMuon = True
)

gemEfficiencyAnalyzerSta = gemEfficiencyAnalyzer.clone(
    muonTag = "gemOfflineDQMStaMuons",
    folder = 'GEM/Efficiency/type2',
    name = 'STA Muon',
    useGlobalMuon = False
)

gemEfficiencyAnalyzerTightGlbSeq = cms.Sequence(
    cms.ignore(gemOfflineDQMTightGlbMuons) *
    gemEfficiencyAnalyzerTightGlb)

gemEfficiencyAnalyzerStaSeq = cms.Sequence(
    cms.ignore(gemOfflineDQMStaMuons) *
    gemEfficiencyAnalyzerSta)
