import FWCore.ParameterSet.Config as cms

from SimGeneral.HepPDTESSource.pythiapdt_cfi import *

# SEQUENCE FOR PAT TRACKS

from RecoMuon.MuonIsolationProducers.muIsolation_cff import *

from PhysicsTools.PatAlgos.recoLayer0.genericTrackCandidates_cff import *
patAODTrackCands.cut = 'pt > 10.'
#from PhysicsTools.PatAlgos.cleaningLayer0.genericTrackCleaner_cfi import *
#allLayer0TrackCands.removeOverlaps = cms.PSet(
#)

  # add in MC match

from PhysicsTools.PatAlgos.mcMatchLayer0.trackMuMatch_cfi import *
trackMuMatch.maxDeltaR = 0.15
trackMuMatch.maxDPtRel = 1.0
trackMuMatch.resolveAmbiguities = False

  # Layer 1 
import PhysicsTools.PatAlgos.producersLayer1.genericParticleProducer_cfi as genericpartproducer_cfi

allLayer1TrackCands = genericpartproducer_cfi.allLayer1GenericParticles.clone(
    src = cms.InputTag("allLayer0TrackCands"),
    isolation = cms.PSet(
      tracker = cms.PSet(
        veto = cms.double(0.015),
        src = cms.InputTag("layer0TrackIsolations","patAODTrackIsoDepositCtfTk"),
        deltaR = cms.double(0.3),
        threshold = cms.double(1.5)
      ),
      ecal = cms.PSet(
        src = cms.InputTag("layer0TrackIsolations","patAODTrackIsoDepositCalByAssociatorTowersecal"),
        deltaR = cms.double(0.3)
      ),
      hcal = cms.PSet(
        src = cms.InputTag("layer0TrackIsolations","patAODTrackIsoDepositCalByAssociatorTowershcal"),
        deltaR = cms.double(0.3)
      ),
    ),
    isoDeposits = cms.PSet(
      tracker = cms.InputTag("layer0TrackIsolations","patAODTrackIsoDepositCtfTk"),
      ecal = cms.InputTag("layer0TrackIsolations","patAODTrackIsoDepositCalByAssociatorTowersecal"),
      hcal = cms.InputTag("layer0TrackIsolations","patAODTrackIsoDepositCalByAssociatorTowershcal")
    ),
    addGenMatch = cms.bool(True),
    genParticleMatch = cms.InputTag("trackMuMatch")
)

from PhysicsTools.PatAlgos.selectionLayer1.trackSelector_cfi import *
selectedLayer1TrackCands.cut = 'pt > 10.'

# SEQUENCE FOR DIMUONS

from ElectroWeakAnalysis.ZReco.dimuons_cfi import *
from ElectroWeakAnalysis.ZReco.dimuonsOneTrack_cfi import *
from ElectroWeakAnalysis.ZReco.dimuonsGlobal_cfi import *
from ElectroWeakAnalysis.ZReco.dimuonsOneStandAloneMuon_cfi import *
from ElectroWeakAnalysis.ZReco.mcTruthForDimuons_cff import *

patLayer0 = cms.Sequence(
    patAODTrackCandSequence*
#    allLayer0TrackCands*
    trackMuMatch*
    patLayer0TrackCandSequence
)

patLayer1 = cms.Sequence(
    allLayer1TrackCands*
    selectedLayer1TrackCands
)

goodMuonRecoForDimuon = cms.Sequence(
    patLayer0*
    patLayer1
)


