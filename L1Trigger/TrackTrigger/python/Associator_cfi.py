# config parameter used by Associator
import FWCore.ParameterSet.Config as cms

Associator_params = cms.PSet (

  MinLayers       = cms.int32 (  4    ), # required number of associated stub layers to a TP to consider it reconstruct-able
  MinLayersPS     = cms.int32 (  0    ), # required number of associated ps stub layers to a TP to consider it reconstruct-able
  MinLayersGood   = cms.int32 (  4    ), # required number of layers a found track has to have in common with a TP to consider it matched
  MinLayersGoodPS = cms.int32 (  0    ), # required number of ps layers a found track has to have in common with a TP to consider it matched
  MaxLayersBad    = cms.int32 (  1    ), # max number of unassociated 2S stubs allowed to still associate TTTrack with TP
  MaxLayersBadPS  = cms.int32 (  0    )  # max number of unassociated PS stubs allowed to still associate TTTrack with TP

)
