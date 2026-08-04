import FWCore.ParameterSet.Config as cms

from DQM.SiTrackerPhase2.Phase2OTMonitorCluster_cfi import Phase2OTMonitorCluster 

clusterMonitorOT = Phase2OTMonitorCluster.clone(
        PositionOfClusters_2S = Phase2OTMonitorCluster.PositionOfClusters_2S.clone(
            switch = cms.bool(False)
        ),
        CrackOverview = Phase2OTMonitorCluster.CrackOverview.clone(
            switch = cms.bool(False)
        )
)
