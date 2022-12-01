import FWCore.ParameterSet.Config as cms

from DQM.SiTrackerPhase2.Phase2OTMonitorVectorHits_cfi import Phase2OTMonitorVectorHits

acceptedVecHitsmon = Phase2OTMonitorVectorHits.clone()

rejectedVecHitsmon = Phase2OTMonitorVectorHits.clone(
    TopFolderName = "TrackerPhase2OTVectorHits/Rejected",
    vechitsSrc = cms.InputTag('siPhase2VectorHits', 'rejected')
)
