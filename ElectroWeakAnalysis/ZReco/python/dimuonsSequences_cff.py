import FWCore.ParameterSet.Config as cms

from SimGeneral.HepPDTESSource.pythiapdt_cfi import *
#  include "PhysicsTools/RecoAlgos/data/goodMuons.cfi"
from PhysicsTools.PatAlgos.cleaningLayer0.muonCleaner_cfi import *
from PhysicsTools.PatAlgos.recoLayer0.muonIsolation_cff import *
from PhysicsTools.PatAlgos.triggerLayer0.muonHLTProducer_cfi import *
from PhysicsTools.PatAlgos.triggerLayer0.muonHLTMatch_cfi import *
from PhysicsTools.PatAlgos.mcMatchLayer0.muonMatch_cfi import *
from PhysicsTools.PatAlgos.producersLayer1.muonProducer_cff import *
from PhysicsTools.RecoAlgos.goodTracks_cfi import *
#
# stand-alone muons are already included in the standard muon collection
#
#  include "PhysicsTools/RecoAlgos/data/goodStandAloneMuonTracks.cfi"
from PhysicsTools.IsolationAlgos.goodMuonIsolations_cfi import *
from PhysicsTools.IsolationAlgos.goodTrackIsolations_cfi import *
from ElectroWeakAnalysis.ZReco.muonIsolations_cfi import *
#  include "PhysicsTools/IsolationAlgos/data/highPtTrackIsolations.cfi"
from ElectroWeakAnalysis.ZReco.dimuons_cfi import *
from ElectroWeakAnalysis.ZReco.dimuonsOneTrack_cfi import *
from ElectroWeakAnalysis.ZReco.dimuonsGlobal_cfi import *
from ElectroWeakAnalysis.ZReco.dimuonsOneStandAloneMuon_cfi import *
from ElectroWeakAnalysis.ZReco.mcTruthForDimuons_cff import *
patLayer0 = cms.Sequence(patAODMuonIsolation*allLayer0Muons*patLayer0MuonIsolation*muonHLTProducer*muonHLTMatch*muonMatch)
patLayer1 = cms.Sequence(layer1Muons)
goodMuonRecoForDimuon = cms.Sequence(patLayer0*patLayer1*goodTracks*goodTrackIsolations*goodMuonIsolations*muonIsolations)
allLayer0Muons.isolation.tracker = cms.PSet(
    veto = cms.double(0.015),
    src = cms.InputTag("patAODMuonIsolations","muIsoDepositTk"),
    deltaR = cms.double(0.3),
    cut = cms.double(3.0),
    threshold = cms.double(1.5)
)
muonMatch.maxDeltaR = 0.15
muonMatch.maxDPtRel = 1.0
muonMatch.resolveAmbiguities = False
selectedLayer1Muons.src = 'allLayer1Muons'
selectedLayer1Muons.cut = 'pt > 0. & abs(eta) < 100.0'
goodMuonIsolations.src = 'selectedLayer1Muons'

