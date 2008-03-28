import FWCore.ParameterSet.Config as cms

from SimGeneral.HepPDTESSource.pythiapdt_cfi import *
from PhysicsTools.RecoAlgos.goodMuons_cfi import *
from PhysicsTools.RecoAlgos.goodTracks_cfi import *
from PhysicsTools.RecoAlgos.goodStandAloneMuonTracks_cfi import *
TagMuons = cms.EDFilter("MuonRefSelector",
    src = cms.InputTag("muons"),
    cut = cms.string('pt > 4')
)

muonRecoForTrackingEff = cms.Sequence(goodMuons+goodTracks+goodStandAloneMuonTracks+TagMuons)

