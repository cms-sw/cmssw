import FWCore.ParameterSet.Config as cms

# -*-TCL-*-
from RecoMuon.MuonIsolationProducers.isoDepositProducerIOBlocks_cff import *
from RecoMuon.MuonIsolationProducers.jetExtractorBlock_cff import *
muIsoDepositJets = cms.EDProducer("MuIsoDepositProducer",
    IOPSet = cms.PSet(
        MIsoDepositViewIOBlock
    ),
    ExtractorPSet = cms.PSet(
        MIsoJetExtractorBlock
    )
)



