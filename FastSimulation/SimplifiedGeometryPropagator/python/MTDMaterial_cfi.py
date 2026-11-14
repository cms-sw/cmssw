import FWCore.ParameterSet.Config as cms

_btlInteractionModels = cms.untracked.vstring(
    "pairProductionBTL",
    "nuclearInteractionBTL",
    "bremsstrahlungBTL",
    "energyLossBTL",
    "multipleScatteringBTL",
    "mtdSimHits",
)
_etlInteractionModels = cms.untracked.vstring(
    "pairProductionETL",
    "nuclearInteractionETL",
    "bremsstrahlungETL",
    "energyLossETL",
    "multipleScatteringETL",
    "mtdSimHits",
)

MTDMaterialBlock = cms.PSet(
    MTDMaterial = cms.PSet(
        maxRadius = cms.untracked.double(150.),
        maxZ      = cms.untracked.double(325.),
        useTrackerRecoGeometryRecord = cms.untracked.bool(False),
        useMTDRecoGeometryRecord     = cms.untracked.bool(True),

        BarrelLayers = cms.VPSet(
            # BTL: LYSO crystal barrel
            cms.PSet(
                radius    = cms.untracked.double(115.8),
                limits    = cms.untracked.vdouble(0.0, 260.),
                thickness = cms.untracked.vdouble(3.29),
                interactionModels = _btlInteractionModels,
            ),
        ),
        EndcapLayers = cms.VPSet(
            # ETL DoubleLayer 1 (disk 1 z=299.5 + disk 2 z=300.7)
            cms.PSet(
                z         = cms.untracked.double(300.1),
                limits    = cms.untracked.vdouble(31., 120.),
                thickness = cms.untracked.vdouble(0.0064),
                interactionModels = _etlInteractionModels,
            ),
            # ETL DoubleLayer 2 (disk 3 z=301.9 + disk 4 z=303.1)
            cms.PSet(
                z         = cms.untracked.double(302.5),
                limits    = cms.untracked.vdouble(31., 120.),
                thickness = cms.untracked.vdouble(0.0064),
                interactionModels = _etlInteractionModels,
            ),
        ),
    )
)
