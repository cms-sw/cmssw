import FWCore.ParameterSet.Config as cms

from PhysicsTools.PatAlgos.recoLayer0.bTagging_cff import *
from PhysicsTools.PatAlgos.recoLayer0.jetTracksCharge_cff import *
from PhysicsTools.PatAlgos.recoLayer0.jetCorrections_cff import *
from PhysicsTools.PatAlgos.mcMatchLayer0.jetMatch_cfi import *
from PhysicsTools.PatAlgos.mcMatchLayer0.jetFlavourId_cff import *
from PhysicsTools.PatAlgos.producersLayer1.jetProducer_cfi import *

## for scheduled mode
makePatJetsTask = cms.Task(
    patJetCorrectionsTask,
    patJetCharge,
    patJetPartonMatch,
    patJetGenJetMatch,
    patJetFlavourIdLegacyTask,
    patJetFlavourIdTask,
    patJets
    )
makePatJets = cms.Sequence(makePatJetsTask)
