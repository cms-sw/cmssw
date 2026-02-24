import FWCore.ParameterSet.Config as cms
from RecoHGCal.TICL.iterativeTICL_cff import ticlIterLabels

trackstersIters = ['keep *_'+iteration+'_*_*' for iteration in ticlIterLabels]

TICL_AOD = cms.PSet(
    outputCommands = cms.untracked.vstring()
)

# RECO content - Includes associations and intermediate steps
TICL_RECO = cms.PSet(
    outputCommands = cms.untracked.vstring(
      trackstersIters +
      [
       'keep *_ticlTrackstersHFNoseTrkEM_*_*',
       'keep *_ticlTrackstersHFNoseEM_*_*',
       'keep *_ticlTrackstersHFNoseTrk_*_*',
       'keep *_ticlTrackstersHFNoseMIP_*_*',
       'keep *_ticlTrackstersHFNoseHAD_*_*',
       'keep *_ticlTrackstersHFNoseMerge_*_*',
       'keep *_ticlCandidate_*_*',
       'keep *_ticlTracksterLinks*_*_*',
       'keep *_pfTICL_*_*',
       'keep CaloParticles_mix_*_*', 
       'keep SimClusters_mix_*_*', 
       'keep *_SimClusterToCaloParticleAssociation*_*_*',
       'keep *_layerClusterSimClusterAssociationProducer_*_*',
       'keep *_layerClusterCaloParticleAssociationProducer_*_*', 
       'keep *_layerClusterSimTracksterAssociationProducer_*_*',
       'keep *_allTrackstersToSimTrackstersAssociations*_*_*'
      ]
    )
)
TICL_RECO.outputCommands.extend(TICL_AOD.outputCommands)

# FEVT Content - Full debug info
TICL_FEVT = cms.PSet(
    outputCommands = cms.untracked.vstring(
      'keep *_ticlSimTracksters_*_*',
      'keep *_ticlSimTICLCandidates_*_*',
      'keep *_ticlSimTrackstersFromCP_*_*',
      'keep *_SimTau*_*_*',
      'keep *_allTrackstersToSimTrackstersAssociations*_*_*'
      )
)
TICL_FEVT.outputCommands.extend(TICL_RECO.outputCommands)

# HLT Content
TICL_FEVTHLT = cms.PSet(
    outputCommands = cms.untracked.vstring(
            ['keep *_hltPfTICL_*_*',
            'keep *_hltTiclTrackstersCLUE3D*_*_*',
            'keep *_hltTiclTracksterLinks*_*_*',
            'keep *_hltTiclCandidate_*_*']
    )
)
TICL_FEVTHLT.outputCommands.extend(TICL_FEVT.outputCommands)


def customiseHGCalOnlyEventContent(process):
    def cleanOutputAndSet(outputModule, ticl_outputCommands):
        outputModule.outputCommands = ['drop *_*_*_*']
        outputModule.outputCommands.extend(ticl_outputCommands)
        outputModule.outputCommands.extend(['keep *_HGCalRecHit_*_*',
                                            'keep *_hgcalMergeLayerClusters_*_*',
                                            'keep CaloParticles_mix_*_*',
                                            'keep SimClusters_mix_*_*',
                                            'keep recoTracks_generalTracks_*_*',
                                            'keep recoTrackExtras_generalTracks_*_*',
                                            'keep SimTracks_g4SimHits_*_*',
                                            'keep SimVertexs_g4SimHits_*_*',
                                            'keep *_layerClusterSimClusterAssociationProducer_*_*',
                                            'keep *_layerClusterCaloParticleAssociationProducer_*_*',
                                            'keep *_randomEngineStateProducer_*_*',
                                            'keep *_layerClusterSimTracksterAssociationProducer_*_*',
                                            'keep *_SimClusterToCaloParticleAssociation*_*_*',
                                            'keep *_simClusterToCaloParticleAssociator*_*_*',
                                            'keep *_SimTau*_*_*',
                                            'keep *_allTrackstersToSimTrackstersAssociations*_*_*'
                                            ])

    if hasattr(process, 'FEVTDEBUGEventContent'):
        cleanOutputAndSet(process.FEVTDEBUGEventContent, TICL_FEVT.outputCommands)
    if hasattr(process, 'FEVTDEBUGHLToutput'):
        cleanOutputAndSet(process.FEVTDEBUGHLToutput, TICL_FEVTHLT.outputCommands)

    return process

