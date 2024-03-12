import FWCore.ParameterSet.Config as cms

# -*-TCL-*-
from RecoMuon.MuonIsolationProducers.isoDepositProducerIOBlocks_cff import *
from RecoMuon.MuonIsolationProducers.caloExtractorByAssociatorBlocks_cff import *
muIsoDepositCalByAssociatorTowers = cms.EDProducer("MuIsoDepositProducer",
    IOPSet = cms.PSet(
        MIsoDepositViewMultiIOBlock
    ),
    ExtractorPSet = cms.PSet(
        MIsoCaloExtractorByAssociatorTowersBlock
    )
)



# foo bar baz
# qWCykviLZ34ZS
# z0wzOjPjUZDGk
