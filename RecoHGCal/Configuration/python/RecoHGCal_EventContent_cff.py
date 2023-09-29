import FWCore.ParameterSet.Config as cms

from RecoHGCal.TICL.iterativeTICL_cff import ticlIterLabelsMerge

trackstersIters = ['keep *_ticlTracksters'+iteration+'_*_*' for iteration in ticlIterLabelsMerge]

#AOD content
TICL_AOD = cms.PSet(
    outputCommands = cms.untracked.vstring()
    )

#RECO content
TICL_RECO = cms.PSet(
    outputCommands = cms.untracked.vstring(
      trackstersIters +
      ['keep *_ticlTrackstersHFNoseTrkEM_*_*',
       'keep *_ticlTrackstersHFNoseEM_*_*',
       'keep *_ticlTrackstersHFNoseTrk_*_*',
       'keep *_ticlTrackstersHFNoseMIP_*_*',
       'keep *_ticlTrackstersHFNoseHAD_*_*',
       'keep *_ticlTrackstersHFNoseMerge_*_*',] +
      ['keep *_pfTICL_*_*']
      )
    )
TICL_RECO.outputCommands.extend(TICL_AOD.outputCommands)

# FEVT Content
TICL_FEVT = cms.PSet(
    outputCommands = cms.untracked.vstring(
      'keep *_ticlSimTracksters_*_*',
      'keep *_ticlSimTrackstersFromCP_*_*',
      )
    )
TICL_FEVT.outputCommands.extend(TICL_RECO.outputCommands)

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
                                            ])

    if hasattr(process, 'FEVTDEBUGEventContent'):
        cleanOutputAndSet(process.FEVTDEBUGEventContent, TICL_FEVT.outputCommands)
    if hasattr(process, 'FEVTDEBUGHLToutput'):
        cleanOutputAndSet(process.FEVTDEBUGHLToutput, TICL_FEVT.outputCommands)

    return process
