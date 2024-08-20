import FWCore.ParameterSet.Config as cms

from RecoHGCal.TICL.iterativeTICL_cff import ticlIterLabelsMerge

trackstersIters = ['keep *_ticlTracksters'+iteration+'_*_*' for iteration in ticlIterLabelsMerge]
trackstersHLTIters = ['keep *_hltTiclTracksters'+iteration+'_*_*' for iteration in ticlIterLabelsMerge]

#AOD content
TICL_AOD = cms.PSet(
    outputCommands = cms.untracked.vstring()
    )

#RECO content
TICL_RECO = cms.PSet(
    outputCommands = cms.untracked.vstring(
      trackstersIters +
      [
       'keep *_ticlTrackstersHFNoseTrkEM_*_*',
       'keep *_ticlTrackstersHFNoseEM_*_*',
       'keep *_ticlTrackstersHFNoseTrk_*_*',
       'keep *_ticlTrackstersHFNoseMIP_*_*',
       'keep *_ticlTrackstersHFNoseHAD_*_*',
       'keep *_ticlTrackstersHFNoseMerge_*_*',] +
      ['keep *_pfTICL_*_*'] +
      ['keep CaloParticles_mix_*_*', 'keep SimClusters_mix_*_*'] +
      ['keep *_layerClusterSimClusterAssociationProducer_*_*','keep *_layerClusterCaloParticleAssociationProducer_*_*', 'keep *_layerClusterSimTracksterAssociationProducer_*_*'] +
      ['keep *_tracksterSimTracksterAssociationLinking_*_*' ,'keep *_tracksterSimTracksterAssociationPR_*_*'] +
      ['keep *_tracksterSimTracksterAssociationLinkingPU_*_*' ,'keep *_tracksterSimTracksterAssociationPRPU_*_*'] +
      ['keep *_tracksterSimTracksterAssociationLinkingbyCLUE3D_*_*', 'keep *_tracksterSimTracksterAssociationPRbyCLUE3D_*_*']
      )
    )

TICLv5_RECO = cms.PSet(
    outputCommands = cms.untracked.vstring(
        [
            'drop *_ticlTracksters*_*_*',
            'keep *_ticlTrackstersCLUE3DHigh_*_*',
            'keep *_ticlTracksterLinks_*_*',
            'keep *_ticlTracksterLinksSuperclustering*_*_*',
            'keep *_ticlCandidate_*_*',
        ]
    )
)


TICL_RECO.outputCommands.extend(TICL_AOD.outputCommands)


# FEVT Content
TICL_FEVT = cms.PSet(
    outputCommands = cms.untracked.vstring(
      'keep *_ticlSimTracksters_*_*',
      'keep *_ticlSimTICLCandidates_*_*',
      'keep *_ticlSimTrackstersFromCP_*_*',
      'keep *_SimTau*_*_*'
      )
    )
TICL_FEVT.outputCommands.extend(TICL_RECO.outputCommands)
TICLv5_FEVT = cms.PSet(
    outputCommands = cms.untracked.vstring(
      'keep *_ticlSimTracksters_*_*',
      'keep *_ticlSimTICLCandidates_*_*',
      'keep *_ticlSimTrackstersFromCP_*_*',
      'keep *_tracksterSimTracksterAssociationLinkingSuperclustering_*_*',
      'keep *_tracksterSimTracksterAssociationPRSuperclustering_*_*', 
      )
    )

TICLv5_FEVT.outputCommands.extend(TICLv5_RECO.outputCommands)

TICL_FEVTHLT = cms.PSet(
    outputCommands = cms.untracked.vstring(
            trackstersHLTIters +
            ['keep *_hltPfTICL_*_*']
    )
)

TICL_FEVTHLT.outputCommands.extend(TICL_FEVT.outputCommands)

TICLv5_FEVTHLT = cms.PSet(
    outputCommands = cms.untracked.vstring(
        [
            'drop *_hltTiclTracksters*_*_*',
            'keep *_hltTiclTrackstersCLUE3D*_*_*',
            'keep *_hltTiclTracksterLinks_*_*',
            'keep *_hltTiclCandidate_*_*',
            'keep *_hltPfTICL_*_*',
        ]
    )
)

TICLv5_FEVTHLT.outputCommands.extend(TICLv5_FEVT.outputCommands)

def customiseHGCalOnlyEventContent(process):
    def cleanOutputAndSet(outputModule, ticl_outputCommads):
        outputModule.outputCommands = ['drop *_*_*_*']
        outputModule.outputCommands.extend(ticl_outputCommads)
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
                                            'keep *_tracksterSimTracksterAssociationLinking_*_*',
                                            'keep *_tracksterSimTracksterAssociationPR_*_*',
                                            'keep *_tracksterSimTracksterAssociationLinkingPU_*_*',
                                            'keep *_tracksterSimTracksterAssociationPRPU_*_*',
                                            'keep *_tracksterSimTracksterAssociationLinkingbyCLUE3D_*_*',
                                            'keep *_tracksterSimTracksterAssociationPRbyCLUE3D_*_*',
                                            ])

    if hasattr(process, 'FEVTDEBUGEventContent'):
        cleanOutputAndSet(process.FEVTDEBUGEventContent, TICL_FEVT.outputCommands)
    if hasattr(process, 'FEVTDEBUGHLToutput'):
        cleanOutputAndSet(process.FEVTDEBUGHLToutput, TICL_FEVTHLT.outputCommands)

    return process



def customiseForTICLv5EventContent(process):
    def cleanOutputAndSet(outputModule, ticl_outputCommands):
        outputModule.outputCommands.extend(ticl_outputCommands)

    if hasattr(process, 'FEVTDEBUGEventContent'):
        cleanOutputAndSet(process.FEVTDEBUGEventContent, TICLv5_FEVT.outputCommands)
    if hasattr(process, 'FEVTDEBUGHLToutput'):
        cleanOutputAndSet(process.FEVTDEBUGHLToutput, TICLv5_FEVTHLT.outputCommands)
    if hasattr(process, 'FEVTEventContent'):
        cleanOutputAndSet(process.FEVTEventContent, TICLv5_FEVT.outputCommands)

    return process

