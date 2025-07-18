import FWCore.ParameterSet.Config as cms

# -*-TCL-*-
from RecoMuon.MuonIsolationProducers.isoDepositProducerIOBlocks_cff import *
from RecoMuon.MuonIsolationProducers.caloExtractorByAssociatorBlocks_cff import *
muIsoDepositCalByAssociatorMixed = cms.EDProducer("MuIsoDepositProducer",
    IOPSet = cms.PSet(
        MIsoDepositViewMultiIOBlock
    ),
    ExtractorPSet = cms.PSet(
        MIsoCaloExtractorByAssociatorMixedBlock
    )
)



