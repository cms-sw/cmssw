import FWCore.ParameterSet.Config as cms
from DQMServices.Core.DQMEDAnalyzer import DQMEDAnalyzer
from RecoMuon.TrackingTools.MuonServiceProxy_cff import MuonServiceProxy


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


gemEfficiencyAnalyzerTight = DQMEDAnalyzer('GEMEfficiencyAnalyzer',
    MuonServiceProxy,
    muonTag = cms.InputTag('gemOfflineDQMTightGlbMuons'),
    recHitTag = cms.InputTag('gemRecHits'),
    residualXCut = cms.double(5.0),
    ptBinning = cms.untracked.vdouble(20. ,30., 40., 50., 60., 70., 80., 90., 100., 120., 140., 200.),
    etaNbins = cms.untracked.int32(7),
    etaLow = cms.untracked.double(1.5),
    etaUp = cms.untracked.double(2.2),
    useGlobalMuon = cms.untracked.bool(True),
    folder = cms.untracked.string('GEM/GEMEfficiency/TightGlobalMuon'),
    logCategory = cms.untracked.string('GEMEfficiencyAnalyzerTight'),
)


gemEfficiencyAnalyzerSTA = gemEfficiencyAnalyzerTight.clone()
gemEfficiencyAnalyzerSTA.muonTag = cms.InputTag("gemOfflineDQMStaMuons")
gemEfficiencyAnalyzerSTA.useGlobalMuon = cms.untracked.bool(False)
gemEfficiencyAnalyzerSTA.folder = cms.untracked.string('GEM/GEMEfficiency/StandaloneMuon')
gemEfficiencyAnalyzerSTA.logCategory = cms.untracked.string('GEMEfficiencyAnalyzerSTA')


from Configuration.Eras.Modifier_phase2_GEM_cff import phase2_GEM
phase2_GEM.toModify(gemEfficiencyAnalyzerTight, etaNbins=cms.untracked.int32(15), etaHigh=cms.untracked.double(3.0))
phase2_GEM.toModify(gemEfficiencyAnalyzerSTA, etaNbins=cms.untracked.int32(15), etaHigh=cms.untracked.double(3.0))


gemEfficiencyAnalyzerTightSeq = cms.Sequence(
    cms.ignore(gemOfflineDQMTightGlbMuons) *
    gemEfficiencyAnalyzerTight)

gemEfficiencyAnalyzerSTASeq = cms.Sequence(
    cms.ignore(gemOfflineDQMStaMuons) *
    gemEfficiencyAnalyzerSTA)
