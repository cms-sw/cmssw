import FWCore.ParameterSet.Config as cms

from SimGeneral.HepPDTESSource.pythiapdt_cfi import *

# SEQUENCE FOR PAT TRACKS

from PhysicsTools.PatAlgos.recoLayer0.genericTrackCandidates_cff import *
patAODTrackCands.cut = 'pt > 10.'
#from PhysicsTools.PatAlgos.cleaningLayer0.genericTrackCleaner_cfi import *
#allLayer0TrackCands.removeOverlaps = cms.PSet(
#    muons = cms.PSet(
#    collection = cms.InputTag("allLayer0Muons"),
#    deltaR = cms.double(0.3),
#    checkRecoComponents = cms.bool(True)
#    )
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

# SEQUENCE FOR PAT MUONS

#from PhysicsTools.PatAlgos.cleaningLayer0.muonCleaner_cfi import *
#allLayer0Muons.isolation.tracker = cms.PSet(
#    veto = cms.double(0.015),
#    src = cms.InputTag("patAODMuonIsolations","muIsoDepositTk"),
#    deltaR = cms.double(0.3),
#    cut = cms.double(3.0),
#    threshold = cms.double(1.5)
#    )

#from PhysicsTools.PatAlgos.recoLayer0.muonIsolation_cff import *
#from PhysicsTools.PatAlgos.triggerLayer0.patTrigProducer_cfi import *
#from PhysicsTools.PatAlgos.triggerLayer0.patTrigMatcher_cfi import *
from PhysicsTools.PatAlgos.mcMatchLayer0.muonMatch_cfi import *
muonMatch.maxDeltaR = 0.15
muonMatch.maxDPtRel = 1.0
muonMatch.resolveAmbiguities = False

from PhysicsTools.PatAlgos.producersLayer1.muonProducer_cff import *
allLayer1Muons.isolation.tracker = cms.PSet(
    veto = cms.double(0.015),
    src = cms.InputTag("layer0MuonIsolations","muIsoDepositTk"),
    deltaR = cms.double(0.3),
    cut = cms.double(3.0),
    threshold = cms.double(1.5)
        )
allLayer1Muons.trigPrimMatch = cms.VInputTag(cms.InputTag("muonTrigMatchHLT1MuonNonIso"))
selectedLayer1Muons.src = 'allLayer1Muons'
selectedLayer1Muons.cut = 'pt > 0. & abs(eta) < 100.0'

from ElectroWeakAnalysis.ZReco.dimuons_cfi import *
from ElectroWeakAnalysis.ZReco.dimuonsOneTrack_cfi import *
from ElectroWeakAnalysis.ZReco.dimuonsGlobal_cfi import *
from ElectroWeakAnalysis.ZReco.dimuonsOneStandAloneMuon_cfi import *
from ElectroWeakAnalysis.ZReco.mcTruthForDimuons_cff import *

patLayer0 = cms.Sequence(
#    patAODMuonIsolation*
#    allLayer0Muons*
    patAODTrackCandSequence*
#    allLayer0TrackCands*
    muonMatch*
    trackMuMatch*
#    patLayer0MuonIsolation*
    patLayer0TrackCandSequence #*
#    patHLT1MuonNonIso*
#    muonTrigMatchHLT1MuonNonIso
    )

patLayer1 = cms.Sequence(
    layer1Muons*
    allLayer1TrackCands*
    selectedLayer1TrackCands )

goodMuonRecoForDimuon = cms.Sequence(
    patLayer0*
    patLayer1 )


