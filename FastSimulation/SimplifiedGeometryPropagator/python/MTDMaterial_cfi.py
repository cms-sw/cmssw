import FWCore.ParameterSet.Config as cms

_mtdMaterialInteractionModels = cms.untracked.vstring(
    "pairProduction",
    "nuclearInteraction",
    "bremsstrahlung",
    "energyLoss",
    "multipleScattering",
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
                interactionModels = _mtdMaterialInteractionModels,
            ),
        ),

        EndcapLayers = cms.VPSet(
            # ETL disk 1
            cms.PSet(
                z         = cms.untracked.double(299.5),
                limits    = cms.untracked.vdouble(31., 120.),
                thickness = cms.untracked.vdouble(0.0032),
                interactionModels = _mtdMaterialInteractionModels,
            ),
            # ETL disk 2
            cms.PSet(
                z         = cms.untracked.double(300.7),
                limits    = cms.untracked.vdouble(31., 120.),
                thickness = cms.untracked.vdouble(0.0032),
                interactionModels = _mtdMaterialInteractionModels,
            ),
            # ETL disk 3
            cms.PSet(
                z         = cms.untracked.double(301.9),
                limits    = cms.untracked.vdouble(31., 120.),
                thickness = cms.untracked.vdouble(0.0032),
                interactionModels = _mtdMaterialInteractionModels,
            ),
            # ETL disk 4
            cms.PSet(
                z         = cms.untracked.double(303.1),
                limits    = cms.untracked.vdouble(31., 120.),
                thickness = cms.untracked.vdouble(0.0032),
                interactionModels = _mtdMaterialInteractionModels,
            ),
        ),
    )
)
