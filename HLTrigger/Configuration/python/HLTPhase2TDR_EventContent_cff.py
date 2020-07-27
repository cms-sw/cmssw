import FWCore.ParameterSet.Config as cms

HLTPhase2TDR = cms.PSet(
    outputCommands = cms.vstring( *(
        'keep *_TTTrackAssociatorFromPixelDigis_*_*',  
        'keep *_TTStubAssociatorFromPixelDigis_*_*',  
        'drop *_TTStubsFromPhase2TrackerDigis_*_HLT',
        'drop Phase2TrackerDigiedmDetSetVectorPhase2TrackerDigiPhase2TrackerDigiedmrefhelperFindForDetSetVectoredmRefTTClusteredmNewDetSetVector_TTClustersFromPhase2TrackerDigis_ClusterInclusive_*',
        'drop Phase2TrackerDigiedmDetSetVectorPhase2TrackerDigiPhase2TrackerDigiedmrefhelperFindForDetSetVectoredmRefTTClusterAssociationMap_TTClusterAssociatorFromPixelDigis_ClusterInclusive_*',
        'drop Phase2TrackerDigiedmDetSetVectorPhase2TrackerDigiPhase2TrackerDigiedmrefhelperFindForDetSetVectoredmRefTTClusterAssociationMap_TTClusterAssociatorFromPixelDigis_ClusterAccepted_*',
        'drop recoPFClusters_particleFlowClusterHGCal__*',
        'drop l1tHGCalTriggerCellBXVector_hgcalVFEProducer_HGCalVFEProcessorSums_*',
        'drop recoHGCalMultiClusters_ticlMultiClustersFromTrackstersMerge__*',
        'drop recoHGCalMultiClusters_ticlMultiClustersFromTrackstersTrk__*',
        'drop Phase2TrackerDigiedmDetSetVectorPhase2TrackerDigiPhase2TrackerDigiedmrefhelperFindForDetSetVectoredmRefTTClusteredmNewDetSetVector_TTClustersFromPhase2TrackerDigis_ClusterInclusive_HLT',
        'drop recoHGCalMultiClusters_ticlMultiClustersFromTrackstersMIP__*',
        'drop recoPFClusters_particleFlowClusterHGCalFromMultiCl__*',
        'drop l1tHGCalClusterBXVector_hgcalBackEndLayer1Producer_HGCalBackendLayer1Processor2DClustering_*',
        'drop Phase2TrackerDigiedmDetSetVectorPhase2TrackerDigiPhase2TrackerDigiedmrefhelperFindForDetSetVectoredmRefTTClusterAssociationMap_TTClusterAssociatorFromPixelDigis_ClusterInclusive_HLT',
        'drop l1tHGCalTowerMapBXVector_hgcalTowerMapProducer_HGCalTowerMapProcessor_*',
        'drop recoGsfTrackExtras_electronGsfTracks__*',
        'drop recoCaloClusters_particleFlowSuperClusterHGCal__*',
        'drop l1tHGCalTriggerCellBXVector_hgcalConcentratorProducer_HGCalConcentratorProcessorSelection_*',
        'drop recoSuperClusters_particleFlowSuperClusterHGCal__*',
        'drop recoGsfTrackExtras_electronGsfTracksFromMultiCl__*',
        'drop recoHGCalMultiClusters_hgcalMultiClusters__*',
        'drop recoGsfTrackExtras_electronGsfTracks__*',
        'drop recoCaloClusters_particleFlowSuperClusterHGCalFromMultiCl__*',
        'drop recoSuperClusters_particleFlowSuperClusterHGCalFromMultiCl__*',
        'drop Phase2TrackerDigiedmDetSetVectorPhase2TrackerDigiPhase2TrackerDigiedmrefhelperFindForDetSetVectoredmRefTTClusteredmNewDetSetVector_TTStubsFromPhase2TrackerDigis_ClusterAccepted_HLT',
        'drop recoElectronSeeds_electronMergedSeedsFromMultiCl__*',
        'drop recoTrackExtras_electronGsfTracks__*',
        'drop Phase2TrackerDigiedmDetSetVectorPhase2TrackerDigiPhase2TrackerDigiedmrefhelperFindForDetSetVectoredmRefTTClusterAssociationMap_TTClusterAssociatorFromPixelDigis_ClusterAccepted_HLT',
        'drop recoHGCalMultiClusters_ticlMultiClustersFromTrackstersHAD__*',
        'drop recoTrackExtras_electronGsfTracksFromMultiCl__*',
        'drop recoGsfElectrons_ecalDrivenGsfElectronsFromMultiCl__*',
        'drop l1tHGCalMulticlusterBXVector_hgcalBackEndLayer2Producer_HGCalBackendLayer2Processor3DClustering_*',
        'drop Phase2TrackerDigiedmDetSetVectorPhase2TrackerDigiPhase2TrackerDigiedmrefhelperFindForDetSetVectoredmRefTTStubAssociationMap_TTStubAssociatorFromPixelDigis_StubAccepted_HLT',
        'drop recoGsfTracks_electronGsfTracks__*',
        'drop CaloTowersSorted_towerMaker__*',
        'drop recoGsfTracks_electronGsfTracksFromMultiCl__*',
        'drop l1tHGCalTowerBXVector_hgcalTowerProducer_HGCalTowerProcessor_*',
        'drop TrackingRecHitsOwned_electronGsfTracks__*',
        'drop recoHGCalMultiClusters_ticlMultiClustersFromTrackstersEM__*',
        'drop *_hltGtStage2Digis_*_HLT',
        'drop *_simBmtfDigis_*_HLT',
        'drop *_simCaloStage2Digis_*_HLT',
        'drop *_simCaloStage2Layer1Digis_*_HLT',
        'drop *_simEmtfDigis_*_HLT',
        'drop *_simGmtStage2Digis_*_HLT',
        'drop *_simGtStage2Digis_*_HLT',
        'drop *_simOmtfDigis_*_HLT',
        
    ) )
)

def extendInputEvtContentForHLTTDR(source):
    if not hasattr(source,"inputCommands"):
        source.inputCommands = cms.untracked.vstring("keep *")
    source.inputCommands.extend(HLTPhase2TDR.outputCommands)        
    source.dropDescendantsOfDroppedBranches = cms.untracked.bool(False)

def extendOutputEvtContentForHLTTDR(output):
    if not hasattr(output,"outputCommands"):
        output.outputCommands = cms.untracked.vstring("keep *")
    output.outputCommands.extend(HLTPhase2TDR.outputCommands)        


