import FWCore.ParameterSet.Config as cms

# -*-SH-*-
from RecoMuon.MuonIsolationProducers.caloExtractorByAssociatorBlocks_cff import *
from RecoMuon.MuonIsolationProducers.trackExtractorBlocks_cff import *
from RecoMuon.MuonIsolationProducers.jetExtractorBlock_cff import *
MIdIsoExtractorPSetBlock = cms.PSet(
    CaloExtractorPSet = cms.PSet(
        MIsoCaloExtractorByAssociatorTowersBlock
    ),
    TrackExtractorPSet = cms.PSet(
        MIsoTrackExtractorBlock
    ),
    JetExtractorPSet = cms.PSet(
        MIsoJetExtractorBlock
    ),
    trackDepositName = cms.string("tracker"),
    ecalDepositName  = cms.string("ecal"),
    hcalDepositName  = cms.string("hcal"),
    hoDepositName  = cms.string("ho"),
    jetDepositName  = cms.string("jets"),

)

