import FWCore.ParameterSet.Config as cms

##____________________________________________________________________________||
from RecoMET.Configuration.RecoMET_cff import *
from RecoMET.Configuration.RecoMET_BeamHaloId_cff import *

hcalnoise.fillTracks = False

CSCHaloData.CosmicMuonLabel = cms.InputTag("muons")

##____________________________________________________________________________||
metrecoCosmics = cms.Sequence(
      caloMet+
      caloMetBE+
      caloMetBEFO+
      muonMETValueMapProducer+
      corMetGlobalMuons+
      caloMetM +
      hcalnoise+
      BeamHaloId
      )

##____________________________________________________________________________||
metrecoCosmics_woBeamHaloId = cms.Sequence(
    caloMet+
    caloMetBE+
    caloMetBEFO+
    muonMETValueMapProducer+
    corMetGlobalMuons+
    caloMetM +
    hcalnoise
    )

##____________________________________________________________________________||
metrecoCosmics_woHcalNoise = cms.Sequence(
    caloMet+
    caloMetBE+
    caloMetBEFO+
    muonMETValueMapProducer+
    corMetGlobalMuons+
    caloMetM +
    BeamHaloId
)

##____________________________________________________________________________||
