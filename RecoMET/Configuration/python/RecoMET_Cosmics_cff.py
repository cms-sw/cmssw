import FWCore.ParameterSet.Config as cms

##____________________________________________________________________________||
from RecoMET.Configuration.RecoMET_cff import *
from RecoMET.Configuration.RecoMET_BeamHaloId_cff import *

tcMet.trackInputTag = 'ctfWithMaterialTracksP5LHCNavigation'
tcMet.isCosmics = True

hcalnoise.fillTracks = False

CSCHaloData.CosmicMuonLabel = cms.InputTag("muons")

##____________________________________________________________________________||
metrecoCosmics = cms.Sequence(
      met+
      metNoHF+
      metHO+
      muonMETValueMapProducer+
      corMetGlobalMuons+
      muonTCMETValueMapProducer+
      tcMet+
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
    muonTCMETValueMapProducer+
    tcMet+
    hcalnoise
    )

##____________________________________________________________________________||
metrecoCosmics_woHcalNoise = cms.Sequence(
    met+
    metNoHF+
    metHO+
    muonMETValueMapProducer+
    corMetGlobalMuons+
    muonTCMETValueMapProducer+
    tcMet+
    BeamHaloId
)

##____________________________________________________________________________||
