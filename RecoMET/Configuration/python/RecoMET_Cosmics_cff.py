import FWCore.ParameterSet.Config as cms

##____________________________________________________________________________||
from RecoMET.Configuration.RecoMET_cff import *
from RecoMET.Configuration.RecoMET_BeamHaloId_cff import *

hcalnoise.fillTracks = False

CSCHaloData.CosmicMuonLabel = "muons"

##____________________________________________________________________________||
metrecoCosmicsTask = cms.Task(
      caloMet,
      caloMetBE,
      caloMetBEFO,
      muonMETValueMapProducer,
      caloMetM,
      hcalnoise,
      BeamHaloIdTask
      )
metrecoCosmics = cms.Sequence(metrecoCosmicsTask)

##____________________________________________________________________________||
metrecoCosmics_woBeamHaloIdTask = cms.Task(
    caloMet,
    caloMetBE,
    caloMetBEFO,
    muonMETValueMapProducer,
    caloMetM,
    hcalnoise
    )
metrecoCosmics_woBeamHaloId = cms.Sequence(metrecoCosmics_woBeamHaloIdTask)

##____________________________________________________________________________||
metrecoCosmics_woHcalNoiseTask = cms.Task(
    caloMet,
    caloMetBE,
    caloMetBEFO,
    muonMETValueMapProducer,
    caloMetM,
    BeamHaloIdTask
)
metrecoCosmics_woHcalNoise = cms.Sequence(metrecoCosmics_woHcalNoiseTask)

##____________________________________________________________________________||
