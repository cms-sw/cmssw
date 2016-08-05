import FWCore.ParameterSet.Config as cms

#Full Event content 
RecoLocalTrackerFEVT = cms.PSet(
    outputCommands = cms.untracked.vstring(
    'keep DetIdedmEDCollection_siStripDigis_*_*',
    'keep DetIdedmEDCollection_siPixelDigis_*_*',
    'keep *_siPixelClusters_*_*', 
    'keep *_siStripClusters_*_*',
    'keep *_clusterSummaryProducer_*_*')
)
#RECO content
RecoLocalTrackerRECO = cms.PSet(
    outputCommands = cms.untracked.vstring(
    'keep DetIdedmEDCollection_siStripDigis_*_*',
    'keep DetIdedmEDCollection_siPixelDigis_*_*',
    'keep *_siPixelClusters_*_*', 
    'keep *_siStripClusters_*_*',
    'keep ClusterSummary_clusterSummaryProducer_*_*')
)
#AOD content
RecoLocalTrackerAOD = cms.PSet(
    outputCommands = cms.untracked.vstring('keep ClusterSummary_clusterSummaryProducer_*_*')
)

from Configuration.StandardSequences.Eras import eras
eras.phase2_tracker.toModify(RecoLocalTrackerFEVT, outputCommands = RecoLocalTrackerFEVT.outputCommands + ['keep *_siPhase2Clusters_*_*',
                                                                                                           'keep *_phase2ITPixelClusters_*_*'] )
eras.phase2_tracker.toModify(RecoLocalTrackerRECO, outputCommands = RecoLocalTrackerRECO.outputCommands + ['keep *_siPhase2Clusters_*_*',
                                                                                                           'keep *_phase2ITPixelClusters_*_*'] )

