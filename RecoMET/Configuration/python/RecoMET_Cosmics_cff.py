import FWCore.ParameterSet.Config as cms
# $Id: RecoMET_EventContent_cff.py,v 1.15 2012/09/04 21:18:33 sakuma Exp $

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
      metNoHFHO+
      calotoweroptmaker+
      metOpt+
      metOptNoHF+
      calotoweroptmakerWithHO+
      metOptHO+metOptNoHFHO+
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
    metNoHFHO+
    calotoweroptmaker+
    metOpt+
    metOptNoHF+
    calotoweroptmakerWithHO+
    metOptHO+metOptNoHFHO+
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
    metNoHFHO+
    calotoweroptmaker+
    metOpt+
    metOptNoHF+
    calotoweroptmakerWithHO+
    metOptHO+metOptNoHFHO+
    muonMETValueMapProducer+
    corMetGlobalMuons+
    muonTCMETValueMapProducer+
    tcMet+
    BeamHaloId
)

##____________________________________________________________________________||
