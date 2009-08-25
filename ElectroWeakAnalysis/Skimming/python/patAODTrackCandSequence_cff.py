import FWCore.ParameterSet.Config as cms

#from SimGeneral.HepPDTESSource.pythiapdt_cfi           import *
from Configuration.StandardSequences.MagneticField_cff import *

patAODTrackCandsUnfiltered = cms.EDProducer("ConcreteChargedCandidateProducer",
    src          = cms.InputTag("generalTracks"),
    particleType = cms.string('mu+')   # to fix mass hypothesis
)

patAODTrackCands = cms.EDFilter("CandViewSelector",
    src = cms.InputTag("patAODTrackCandsUnfiltered"),
    cut = cms.string('pt > 10')
)


from RecoMuon.MuonIsolationProducers.muIsolation_cff import *
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

# sequence to run on AOD before PAT cleaners
patAODTrackCandSequence = cms.Sequence(
        patAODTrackCandsUnfiltered *
        patAODTrackCands *
        patAODTrackIsoDepositCtfTk *
        patAODTrackIsoDepositCalByAssociatorTowers
)

