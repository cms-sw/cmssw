import FWCore.ParameterSet.Config as cms

#AOD content
RecoLocalTrackerAOD = cms.PSet(
    outputCommands = cms.untracked.vstring('keep ClusterSummary_clusterSummaryProducer_*_*')
)
from Configuration.ProcessModifiers.egamma_lowPt_exclusive_cff import egamma_lowPt_exclusive
egamma_lowPt_exclusive.toModify( RecoLocalTrackerAOD,
    outputCommands = RecoLocalTrackerAOD.outputCommands + ['keep *_siPixelRecHits_*_*', 
							   'keep *_siPixelClusters_*_*'])
#RECO content
RecoLocalTrackerRECO = cms.PSet(
    outputCommands = cms.untracked.vstring(
    'keep DetIds_siStripDigis_*_*',
    'keep DetIdedmEDCollection_siPixelDigis_*_*',
    'keep PixelFEDChanneledmNewDetSetVector_siPixelDigis_*_*',
    'keep *_siPixelClusters_*_*', 
    'keep *_siStripClusters_*_*')
)
RecoLocalTrackerRECO.outputCommands.extend(RecoLocalTrackerAOD.outputCommands)

from Configuration.Eras.Modifier_phase2_tracker_cff import phase2_tracker
phase2_tracker.toModify(RecoLocalTrackerRECO, outputCommands = RecoLocalTrackerRECO.outputCommands + ['keep *_siPhase2Clusters_*_*'] )

#Full Event content 
RecoLocalTrackerFEVT = cms.PSet(
    outputCommands = cms.untracked.vstring('keep *_clusterSummaryProducer_*_*')
)
RecoLocalTrackerFEVT.outputCommands.extend(RecoLocalTrackerRECO.outputCommands)

phase2_tracker.toModify(RecoLocalTrackerFEVT, outputCommands = RecoLocalTrackerFEVT.outputCommands + ['keep *_siPhase2Clusters_*_*'] )
