import FWCore.ParameterSet.Config as cms

clusterAnalysisFilter = cms.EDFilter("ClusterAnalysisFilter",
    sources,
    ClusterInModuleSelector = cms.PSet(
        On = cms.bool(True),
        SkipModules = cms.vuint32(), ##Modules to be skipped in applying this cut

        Accept = cms.bool(True),
        ModulesToLookIn = cms.vuint32(),
        SubDetToLookIn = cms.vuint32(2, 3, 4, 5), ##  TIB=2, TOB=3, TID=4, TEC=5,

        #vuint32 LayerToBeExcluded = {}         #to be implemented
        ClusterConditions = cms.PSet(
            minWidth = cms.double(0.0),
            maxStoN = cms.double(2000.0),
            minStoN = cms.double(10.0),
            maxWidth = cms.double(200.0)
        )
    ),
    TriggerSelector = cms.PSet(
        On = cms.bool(False),
        RBC1 = cms.bool(True),
        RBC2 = cms.bool(True),
        CSC = cms.bool(True),
        RPCTB = cms.bool(True),
        DT = cms.bool(True)
    ),
    TrackNumberSelector = cms.PSet(
        On = cms.bool(False),
        minNTracks = cms.int32(0), ## min <= value < max

        maxNTracks = cms.int32(100)
    ),
    ClusterNumberSelector = cms.PSet(
        On = cms.bool(False),
        maxNClus = cms.int32(1000),
        minNClus = cms.int32(6) ## min <= value < max

    )
)


