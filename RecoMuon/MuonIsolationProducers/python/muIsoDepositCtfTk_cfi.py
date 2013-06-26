import FWCore.ParameterSet.Config as cms

# -*-TCL-*-
from RecoMuon.MuonIsolationProducers.isoDepositProducerIOBlocks_cff import *
from RecoMuon.MuonIsolationProducers.trackExtractorBlocks_cff import *
muIsoDepositCtfTk = cms.EDProducer("MuIsoDepositProducer",
    IOPSet = cms.PSet(
        MIsoDepositViewIOBlock
    ),
    ExtractorPSet = cms.PSet(
        MIsoTrackExtractorCtfBlock
    )
)



