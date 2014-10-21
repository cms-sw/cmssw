import FWCore.ParameterSet.Config as cms

##____________________________________________________________________________||
from RecoMET.Configuration.RecoMET_cff import *
from RecoMET.Configuration.RecoMET_BeamHaloId_cff import *

hcalnoise.fillTracks = False

CSCHaloData.CosmicMuonLabel = cms.InputTag("muons")

##____________________________________________________________________________||
metrecoCosmics = cms.Sequence(
      met+
      metNoHF+
      metHO+
      muonMETValueMapProducer+
      corMetGlobalMuons+
      hcalnoise+
      BeamHaloId
      )

##____________________________________________________________________________||
metrecoCosmics_woBeamHaloId = cms.Sequence(
    met+
    metNoHF+
    metHO+
    muonMETValueMapProducer+
    corMetGlobalMuons+
    hcalnoise
    )

##____________________________________________________________________________||
metrecoCosmics_woHcalNoise = cms.Sequence(
    met+
    metNoHF+
    metHO+
    muonMETValueMapProducer+
    corMetGlobalMuons+
    BeamHaloId
)

##____________________________________________________________________________||
