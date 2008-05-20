import FWCore.ParameterSet.Config as cms

# -*-TCL-*-
from RecoMuon.MuonIsolationProducers.isoDepositProducerIOBlocks_cff import *
from RecoMuon.MuonIsolationProducers.caloExtractorBlocks_cff import *
muIsoDepositCal = cms.EDProducer("MuIsoDepositProducer",
    IOPSet = cms.PSet(
        MIsoDepositViewIOBlock
    ),
    ExtractorPSet = cms.PSet(
        MIsoCaloExtractorHLTBlock
    )
)



