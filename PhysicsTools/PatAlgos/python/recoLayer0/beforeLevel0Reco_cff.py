import FWCore.ParameterSet.Config as cms

from TrackingTools.TransientTrack.TransientTrackBuilder_cfi import *
from MagneticField.Engine.volumeBasedMagneticField_cfi import *
from PhysicsTools.PatAlgos.recoLayer0.jetCorrFactors_cfi import *
from PhysicsTools.PatAlgos.recoLayer0.jetMETCorrections_cff import *
from PhysicsTools.PatAlgos.recoLayer0.electronIsolation_cff import *
from PhysicsTools.PatAlgos.recoLayer0.photonIsolation_cff import *
from PhysicsTools.PatAlgos.recoLayer0.muonIsolation_cff import *
patBeforeLevel0Reco_withoutPFTau = cms.Sequence(jetCorrFactors*patAODElectronIsolation*patAODPhotonIsolation*patAODMuonIsolation*patJetMETCorrections)
patBeforeLevel0Reco = cms.Sequence(patBeforeLevel0Reco_withoutPFTau)

