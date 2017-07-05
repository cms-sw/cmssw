import FWCore.ParameterSet.Config as cms

# Unprescale HLT_MET and HLT_SinglePhoton triggers
import HLTrigger.HLTfilters.hltHighLevel_cfi
hltMonopole = HLTrigger.HLTfilters.hltHighLevel_cfi.hltHighLevel.clone()
hltMonopole.TriggerResultsTag = cms.InputTag( "TriggerResults", "", "HLT" )
hltMonopole.HLTPaths = cms.vstring(
    "HLT_Photon175_v*",
    "HLT_PFMET300_v*",
    "HLT_PFMET170_HBHE_BeamHaloCleaned_v*"
)
hltMonopole.throw = False
hltMonopole.andOr = True

# selection of valid vertex
#primaryVertexFilterForZMM = cms.EDFilter("VertexSelector",
#    src = cms.InputTag("offlinePrimaryVertices"),
#    cut = cms.string("!isFake && abs(z) <= 25 && position.Rho <= 2"), 
#    filter = cms.bool(True),   # otherwise it won't filter the events
#    )

#Data
from Configuration.EventContent.EventContent_cff import AODEventContent
EXOMonopoleAODContent = AODEventContent.clone()
# Barrel Cluster
EXOMonopoleAODContent.outputCommands.append('keep *_hybridSuperClusters_*_*')
# Endcap Cluster
EXOMonopoleAODContent.outputCommands.append('keep *_multi5x5SuperClusters_multi5x5EndcapSuperClusters_*')
EXOMonopoleAODContent.outputCommands.append('keep *_multi5x5SuperClusters_uncleanOnlyMulti5x5EndcapBasicClusters_*')
EXOMonopoleAODContent.outputCommands.append('keep *_multi5x5SuperClusters_uncleanOnlyMulti5x5EndcapSuperClusters_*')
#
EXOMonopoleAODContent.outputCommands.append('keep *_siStripClusters_*_*')
EXOMonopoleAODContent.outputCommands.append('keep *_siPixelClusters_*_*')
EXOMonopoleAODContent.outputCommands.append('drop *_generalTracks_*_*')
EXOMonopoleAODContent.outputCommands.append('keep *_generalTracks_*_*')
EXOMonopoleAODContent.outputCommands.append('drop *_generalTracks_QualityMasks_*')
EXOMonopoleAODContent.outputCommands.append('keep *_ecalRecHit_EcalRecHitsEB_*')
EXOMonopoleAODContent.outputCommands.append('keep *_ecalRecHit_EcalRecHitsEE_*')
EXOMonopoleAODContent.outputCommands.append('keep *_hbhereco_*_*')

##MC
#from Configuration.EventContent.EventContent_cff import AODSIMEventContent
#EXOMonopoleAODSIMContent = AODSIMEventContent.clone()
## Barrel Cluster
#EXOMonopoleAODSIMContent.outputCommands.append('keep *_hybridSuperClusters_*_*')
## Endcap Cluster
#EXOMonopoleAODSIMContent.outputCommands.append('keep *_multi5x5SuperClusters_multi5x5EndcapSuperClusters_*')
#EXOMonopoleAODSIMContent.outputCommands.append('keep *_multi5x5SuperClusters_uncleanOnlyMulti5x5EndcapBasicClusters_*')
#EXOMonopoleAODSIMContent.outputCommands.append('keep *_multi5x5SuperClusters_uncleanOnlyMulti5x5EndcapSuperClusters_*')
##
#EXOMonopoleAODSIMContent.outputCommands.append('keep *_siStripClusters_*_*')
#EXOMonopoleAODSIMContent.outputCommands.append('keep *_siPixelClusters_*_*')
#EXOMonopoleAODSIMContent.outputCommands.append('drop *_generalTracks_*_*')
#EXOMonopoleAODSIMContent.outputCommands.append('keep *_generalTracks_*_*')
#EXOMonopoleAODSIMContent.outputCommands.append('drop *_generalTracks_QualityMasks_*')
#EXOMonopoleAODSIMContent.outputCommands.append('keep *_ecalRecHit_EcalRecHitsEB_*')
#EXOMonopoleAODSIMContent.outputCommands.append('keep *_ecalRecHit_EcalRecHitsEE_*')
#EXOMonopoleAODSIMContent.outputCommands.append('keep *_hbhereco_*_*')
#EXOMonopoleAODSIMContent.outputCommands.append('keep *_generatorSmeared_*_*')
#EXOMonopoleAODSIMContent.outputCommands.append('keep *_g4SimHits_*_*')

# monopole skim sequence
EXOMonopoleSkimSequence = cms.Sequence(
    hltMonopole
    )
