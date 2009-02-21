import FWCore.ParameterSet.Config as cms

from PhysicsTools.PatAlgos.recoLayer0.electronIsolation_cff import *
from PhysicsTools.PatAlgos.recoLayer0.photonIsolation_cff import *
from PhysicsTools.PatAlgos.recoLayer0.muonIsolation_cff import *
from PhysicsTools.PatAlgos.recoLayer0.jetMETCorrections_cff import *
from PhysicsTools.PatAlgos.recoLayer0.bTagging_cff import *
from PhysicsTools.PatAlgos.recoLayer0.tauDiscriminators_cff import *
from PhysicsTools.PatAlgos.recoLayer0.tauIsolation_cff import *
from PhysicsTools.PatAlgos.recoLayer0.pfCandidateIsoDepositSelection_cff import *
from TrackingTools.TransientTrack.TransientTrackBuilder_cfi import *  # needed for the MET

patBeforeLevel0Reco_withoutPFTau = cms.Sequence(
    patAODTauDiscrimination *  # This stays here even if it's withoutPFTau
    patAODBTagging *
    patAODElectronIsolation *
    patAODPhotonIsolation *
    patAODMuonIsolation *
    patAODJetMETCorrections *
    patAODPFCandidateIsoDepositSelection
)

patBeforeLevel0Reco = cms.Sequence(
    patBeforeLevel0Reco_withoutPFTau *
    patPFTauIsolation *
    patAODPFTauIsolation
)

