import FWCore.ParameterSet.Config as cms

from SimGeneral.HepPDTESSource.pythiapdt_cfi           import *
from Configuration.StandardSequences.MagneticField_cff import *

patAODTrackCandsUnfiltered = cms.EDProducer("ConcreteChargedCandidateProducer",
    src          = cms.InputTag("generalTracks"),
    particleType = cms.string('mu+')   # to fix mass hypothesis
)

patAODTrackCands = cms.EDFilter("CandViewSelector",
    src = cms.InputTag("patAODTrackCandsUnfiltered"),
    cut = cms.string('pt > 15')
)

## Configure tracker isolation
from RecoMuon.MuonIsolationProducers.trackExtractorBlocks_cff import MIsoTrackExtractorCtfBlock
patAODTrackIsoDepositCtfTk = cms.EDProducer("CandIsoDepositProducer",
    src                  = cms.InputTag("patAODTrackCands"),
    trackType            = cms.string('best'),
    MultipleDepositsFlag = cms.bool(False),
    ExtractorPSet        = cms.PSet( MIsoTrackExtractorCtfBlock )
)

## Configure calorimetric isolation
from RecoMuon.MuonIsolationProducers.caloExtractorByAssociatorBlocks_cff import MIsoCaloExtractorByAssociatorTowersBlock
patAODTrackIsoDepositCalByAssociatorTowers = cms.EDProducer("CandIsoDepositProducer",
    src                  = cms.InputTag("patAODTrackCands"),
    trackType            = cms.string('best'),
    MultipleDepositsFlag = cms.bool(True),
    ExtractorPSet        = cms.PSet( MIsoCaloExtractorByAssociatorTowersBlock )
)

## Select isolation labels to use
patAODTrackIsolationLabels = cms.VInputTag(
   #cms.InputTag("patAODTrackIsoDepositCalByAssociatorTowers","ecal"), 
   #cms.InputTag("patAODTrackIsoDepositCalByAssociatorTowers","hcal"), 
   #cms.InputTag("patAODTrackIsoDepositCalByAssociatorTowers","ho"), 
    cms.InputTag("patAODTrackIsoDepositCtfTk")
)

# Isolation converter module
patAODTrackIsolations = cms.EDFilter("MultipleIsoDepositsToValueMaps",
    collection   = cms.InputTag("patAODTrackCands"),
    associations = patAODTrackIsolationLabels
)

# Isolation re-keying to clean layer 0 output collection
layer0TrackIsolations = cms.EDFilter("CandManyValueMapsSkimmerIsoDeposits",
    collection   = cms.InputTag("allLayer0TrackCands"),
    backrefs     = cms.InputTag("allLayer0TrackCands"),
    commonLabel  = cms.InputTag("patAODTrackIsolations"),
    associations = patAODTrackIsolationLabels
)

# sequence to run on AOD before PAT cleaners
patAODTrackCandSequence = cms.Sequence(
        patAODTrackCandsUnfiltered *
        patAODTrackCands *
        patAODTrackIsoDepositCalByAssociatorTowers *
        patAODTrackIsoDepositCtfTk *
        patAODTrackIsolations
)

# sequence to run at end of layer 0 
patLayer0TrackCandSequence = cms.Sequence( layer0TrackIsolations )

