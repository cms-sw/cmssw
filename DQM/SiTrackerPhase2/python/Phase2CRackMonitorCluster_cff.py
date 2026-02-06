import FWCore.ParameterSet.Config as cms

from DQM.SiTrackerPhase2.Phase2OTMonitorCluster_cfi import Phase2OTMonitorCluster as _Phase2OTMonitorCluster

clusterMonitorCRACK = _Phase2OTMonitorCluster.clone(
    PositionOfClusters_2S = _Phase2OTMonitorCluster.PositionOfClusters_2S.clone(
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
    PositionOfClusters_2SLadder = _Phase2OTMonitorCluster.PositionOfClusters_2SLadder.clone(
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
    CrackOverview = _Phase2OTMonitorCluster.CrackOverview.clone(
    name = cms.string('Crack_Overview_clusters'),
    title = cms.string('Crack_Overview_clusters;Module;Layer'),
    xmin = cms.double(0.0),
    xmax = cms.double(13.0),
    ymin = cms.double(0.0),
    ymax = cms.double(7.5),
    switch = cms.bool(True)
    )

    #TopFolderName = cms.string('TrackerPhase2OTCluster'),
    #clusterSrc = cms.InputTag('siPhase2Clusters'),
    #mightGet = cms.optional.untracked.vstring
)
