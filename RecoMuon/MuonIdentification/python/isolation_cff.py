import FWCore.ParameterSet.Config as cms

# -*-SH-*-
from RecoMuon.MuonIsolationProducers.caloExtractorByAssociatorBlocks_cff import *
from RecoMuon.MuonIsolationProducers.trackExtractorBlocks_cff import *
MIdIsoExtractorPSetBlock = cms.PSet(
    CaloExtractorPSet = cms.PSet(
        MIsoCaloExtractorByAssociatorTowersBlock
    ),
    TrackExtractorPSet = cms.PSet(
        MIsoTrackExtractorBlock
    )
)

