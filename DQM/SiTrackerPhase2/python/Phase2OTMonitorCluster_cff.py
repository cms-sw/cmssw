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
clusterMonitorCRACK = Phase2OTMonitorCluster.clone(
    # Histograms that are usually set to switch = False in full tracker
    PositionOfClusters_2S = Phase2OTMonitorCluster.PositionOfClusters_2S.clone(
        name = cms.string('PositionOfClusters_2S_module'),
        title = cms.string('PositionsOfClusters_2S_module;Strip;Half-module;'),
        NxBins = cms.int32(1016),
        xmin = cms.double(0.5),
        xmax = cms.double(1016.5),
        NyBins = cms.int32(5),
        ymin = cms.double(-2.5),
        ymax = cms.double(2.5),
        switch = cms.bool(True)
    ),
    CrackOverview = Phase2OTMonitorCluster.CrackOverview.clone(
        name = cms.string('Crack_Overview_clusters'),
        title = cms.string('Crack_Overview_clusters;Module;Layer'),
        xmin = cms.double(0.0),
        xmax = cms.double(13.0),
        ymin = cms.double(0.0),
        ymax = cms.double(7.5),
        switch = cms.bool(True)
    ),
    # Changes to x/y ranges for CRACK readability
    PositionOfClusters_2SLadder = Phase2OTMonitorCluster.PositionOfClusters_2SLadder.clone(
        name = cms.string('PositionOfClusters_2S_Ladder'),
        title = cms.string('PositionsOfClusters_2S_Ladder;Module;Half-module;'),
        NxBins = cms.int32(12),
        xmin = cms.double(0.5),
        xmax = cms.double(12.5),
        NyBins = cms.int32(5),
        ymin = cms.double(-2.5),
        ymax = cms.double(2.5),
        switch = cms.bool(True)
    ),
    GlobalPositionXY_S = Phase2OTMonitorCluster.GlobalPositionXY_S.clone(
        xmin = cms.double(-7.0),
        xmax = cms.double(7.0),
        ymin = cms.double(-10.0),
        ymax = cms.double(50.0)
    ),
    GlobalPositionRZ_S = Phase2OTMonitorCluster.GlobalPositionRZ_S.clone(
        xmin = cms.double(-70.0),
        xmax = cms.double(70.0),
        ymin = cms.double(0.0),
        ymax = cms.double(60.0)
    ),
    NClustersLayer_S = Phase2OTMonitorCluster.NClustersLayer_S.clone(
        xmax = cms.double(50.0),
        NxBins = cms.int32(50)
    )
    #clusterSrc = cms.InputTag('siPhase2Clusters'),
)
