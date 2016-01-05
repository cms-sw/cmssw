import FWCore.ParameterSet.Config as cms

# -----------------------------------------------------------------------
# General settings common to all scenarios
MisalignmentScenarioSettings = cms.PSet(
    setRotations = cms.bool(True),
    setTranslations = cms.bool(True),
    seed = cms.int32(1234567),
    distribution = cms.string('gaussian'),
    setError = cms.bool(False)
)

MisalignmentScenario_5mu = cms.PSet(
    MisalignmentScenarioSettings,
    scale = cms.double(0.01), # shifts in 100um

    TPBHalfBarrels = cms.PSet(
        DetUnits = cms.PSet(
            dXlocal = cms.double(0.05),
            dYlocal = cms.double(0.05),
        ),
    ),

    TIBHalfBarrels = cms.PSet(
        DetUnits = cms.PSet(
            dXlocal = cms.double(0.05),
        ),
    ),

    TOBHalfBarrels = cms.PSet(
        DetUnits = cms.PSet(
            dXlocal = cms.double(0.05),
        ),
    ),

    TPEEndcaps = cms.PSet(
        DetUnits = cms.PSet(
            dXlocal = cms.double(0.05),
            dYlocal = cms.double(0.05),
        ),
    ),

    TIDEndcaps = cms.PSet(
        DetUnits = cms.PSet(
            dXlocal = cms.double(0.05),
        ),
    ),

    TECEndcaps = cms.PSet(
        DetUnits = cms.PSet(
            dXlocal = cms.double(0.05),
        ),
    ),
)

MisalignmentScenario_10mu = cms.PSet(
    MisalignmentScenarioSettings,
    scale = cms.double(0.01), # shifts in 100um

    TPBHalfBarrels = cms.PSet(
        DetUnits = cms.PSet(
            dXlocal = cms.double(0.1),
            dYlocal = cms.double(0.1),
        ),
    ),

    TIBHalfBarrels = cms.PSet(
        DetUnits = cms.PSet(
            dXlocal = cms.double(0.1),
        ),
    ),

    TOBHalfBarrels = cms.PSet(
        DetUnits = cms.PSet(
            dXlocal = cms.double(0.1),
        ),
    ),

    TPEEndcaps = cms.PSet(
        DetUnits = cms.PSet(
            dXlocal = cms.double(0.1),
            dYlocal = cms.double(0.1),
        ),
    ),

    TIDEndcaps = cms.PSet(
        DetUnits = cms.PSet(
            dXlocal = cms.double(0.1),
        ),
    ),

    TECEndcaps = cms.PSet(
        DetUnits = cms.PSet(
            dXlocal = cms.double(0.1),
        ),
    ),
)

MisalignmentScenario_15mu = cms.PSet(
    MisalignmentScenarioSettings,
    scale = cms.double(0.01), # shifts in 100um

    TPBHalfBarrels = cms.PSet(
        DetUnits = cms.PSet(
            dXlocal = cms.double(0.15),
            dYlocal = cms.double(0.15),
        ),
    ),

    TIBHalfBarrels = cms.PSet(
        DetUnits = cms.PSet(
            dXlocal = cms.double(0.15),
        ),
    ),

    TOBHalfBarrels = cms.PSet(
        DetUnits = cms.PSet(
            dXlocal = cms.double(0.15),
        ),
    ),

    TPEEndcaps = cms.PSet(
        DetUnits = cms.PSet(
            dXlocal = cms.double(0.15),
            dYlocal = cms.double(0.15),
        ),
    ),

    TIDEndcaps = cms.PSet(
        DetUnits = cms.PSet(
            dXlocal = cms.double(0.15),
        ),
    ),

    TECEndcaps = cms.PSet(
        DetUnits = cms.PSet(
            dXlocal = cms.double(0.15),
        ),
    ),
)

MisalignmentScenario_20mu = cms.PSet(
    MisalignmentScenarioSettings,
    scale = cms.double(0.01), # shifts in 100um

    TPBHalfBarrels = cms.PSet(
        DetUnits = cms.PSet(
            dXlocal = cms.double(0.2),
            dYlocal = cms.double(0.2),
        ),
    ),

    TIBHalfBarrels = cms.PSet(
        DetUnits = cms.PSet(
            dXlocal = cms.double(0.2),
        ),
    ),

    TOBHalfBarrels = cms.PSet(
        DetUnits = cms.PSet(
            dXlocal = cms.double(0.2),
        ),
    ),

    TPEEndcaps = cms.PSet(
        DetUnits = cms.PSet(
            dXlocal = cms.double(0.2),
            dYlocal = cms.double(0.2),
        ),
    ),

    TIDEndcaps = cms.PSet(
        DetUnits = cms.PSet(
            dXlocal = cms.double(0.2),
        ),
    ),

    TECEndcaps = cms.PSet(
        DetUnits = cms.PSet(
            dXlocal = cms.double(0.2),
        ),
    ),
)

MisalignmentScenario_20muGlobalX = cms.PSet(
    MisalignmentScenarioSettings,
    scale = cms.double(0.01), # shifts in 100um

    TPBHalfBarrels = cms.PSet(
        DetUnits = cms.PSet(
            dX = cms.double(0.2),
        ),
    ),

    TIBHalfBarrels = cms.PSet(
        DetUnits = cms.PSet(
            dX = cms.double(0.2),
        ),
    ),

    TOBHalfBarrels = cms.PSet(
        DetUnits = cms.PSet(
            dX = cms.double(0.2),
        ),
    ),

    TPEEndcaps = cms.PSet(
        DetUnits = cms.PSet(
            dX = cms.double(0.2),
        ),
    ),

    TIDEndcaps = cms.PSet(
        DetUnits = cms.PSet(
            dX = cms.double(0.2),
        ),
    ),

    TECEndcaps = cms.PSet(
        DetUnits = cms.PSet(
            dX = cms.double(0.2),
        ),
    ),
)

MisalignmentScenario_20muLocalX = cms.PSet(
    MisalignmentScenarioSettings,
    scale = cms.double(0.01), # shifts in 100um

    TPBHalfBarrels = cms.PSet(
        DetUnits = cms.PSet(
            dXlocal = cms.double(0.2),
        ),
    ),

    TIBHalfBarrels = cms.PSet(
        DetUnits = cms.PSet(
            dXlocal = cms.double(0.2),
        ),
    ),

    TOBHalfBarrels = cms.PSet(
        DetUnits = cms.PSet(
            dXlocal = cms.double(0.2),
        ),
    ),

    TPEEndcaps = cms.PSet(
        DetUnits = cms.PSet(
            dXlocal = cms.double(0.2),
        ),
    ),

    TIDEndcaps = cms.PSet(
        DetUnits = cms.PSet(
            dXlocal = cms.double(0.2),
        ),
    ),

    TECEndcaps = cms.PSet(
        DetUnits = cms.PSet(
            dXlocal = cms.double(0.2),
        ),
    ),
)

MisalignmentScenario_20muGlobalY = cms.PSet(
    MisalignmentScenarioSettings,
    scale = cms.double(0.01), # shifts in 100um

    TPBHalfBarrels = cms.PSet(
        DetUnits = cms.PSet(
            dY = cms.double(0.2),
        ),
    ),

    TIBHalfBarrels = cms.PSet(
        DetUnits = cms.PSet(
            dY = cms.double(0.2),
        ),
    ),

    TOBHalfBarrels = cms.PSet(
        DetUnits = cms.PSet(
            dY = cms.double(0.2),
        ),
    ),

    TPEEndcaps = cms.PSet(
        DetUnits = cms.PSet(
            dY = cms.double(0.2),
        ),
    ),

    TIDEndcaps = cms.PSet(
        DetUnits = cms.PSet(
            dY = cms.double(0.2),
        ),
    ),

    TECEndcaps = cms.PSet(
        DetUnits = cms.PSet(
            dY = cms.double(0.2),
        ),
    ),
)

MisalignmentScenario_20muLocalY = cms.PSet(
    MisalignmentScenarioSettings,
    scale = cms.double(0.01), # shifts in 100um

    TPBHalfBarrels = cms.PSet(
        DetUnits = cms.PSet(
            dYlocal = cms.double(0.2),
        ),
    ),

    TIBHalfBarrels = cms.PSet(
        DetUnits = cms.PSet(
            dYlocal = cms.double(0.2),
        ),
    ),

    TOBHalfBarrels = cms.PSet(
        DetUnits = cms.PSet(
            dYlocal = cms.double(0.2),
        ),
    ),

    TPEEndcaps = cms.PSet(
        DetUnits = cms.PSet(
            dYlocal = cms.double(0.2),
        ),
    ),

    TIDEndcaps = cms.PSet(
        DetUnits = cms.PSet(
            dYlocal = cms.double(0.2),
        ),
    ),

    TECEndcaps = cms.PSet(
        DetUnits = cms.PSet(
            dYlocal = cms.double(0.2),
        ),
    ),
)

MisalignmentScenario_20muGlobalZ = cms.PSet(
    MisalignmentScenarioSettings,
    scale = cms.double(0.01), # shifts in 100um

    TPBHalfBarrels = cms.PSet(
        DetUnits = cms.PSet(
            dZ = cms.double(0.2),
        ),
    ),

    TIBHalfBarrels = cms.PSet(
        DetUnits = cms.PSet(
            dZ = cms.double(0.2),
        ),
    ),

    TOBHalfBarrels = cms.PSet(
        DetUnits = cms.PSet(
            dZ = cms.double(0.2),
        ),
    ),

    TPEEndcaps = cms.PSet(
        DetUnits = cms.PSet(
            dZ = cms.double(0.2),
        ),
    ),

    TIDEndcaps = cms.PSet(
        DetUnits = cms.PSet(
            dZ = cms.double(0.2),
        ),
    ),

    TECEndcaps = cms.PSet(
        DetUnits = cms.PSet(
            dZ = cms.double(0.2),
        ),
    ),
)

MisalignmentScenario_20muLocalZ = cms.PSet(
    MisalignmentScenarioSettings,
    scale = cms.double(0.01), # shifts in 100um

    TPBHalfBarrels = cms.PSet(
        DetUnits = cms.PSet(
            dZlocal = cms.double(0.2),
        ),
    ),

    TIBHalfBarrels = cms.PSet(
        DetUnits = cms.PSet(
            dZlocal = cms.double(0.2),
        ),
    ),

    TOBHalfBarrels = cms.PSet(
        DetUnits = cms.PSet(
            dZlocal = cms.double(0.2),
        ),
    ),

    TPEEndcaps = cms.PSet(
        DetUnits = cms.PSet(
            dZlocal = cms.double(0.2),
        ),
    ),

    TIDEndcaps = cms.PSet(
        DetUnits = cms.PSet(
            dZlocal = cms.double(0.2),
        ),
    ),

    TECEndcaps = cms.PSet(
        DetUnits = cms.PSet(
            dZlocal = cms.double(0.2),
        ),
    ),
)

MisalignmentScenario_20muGlobalXY = cms.PSet(
    MisalignmentScenarioSettings,
    scale = cms.double(0.01), # shifts in 100um

    TPBHalfBarrels = cms.PSet(
        DetUnits = cms.PSet(
            dX = cms.double(0.2),
            dY = cms.double(0.2),
        ),
    ),

    TIBHalfBarrels = cms.PSet(
        DetUnits = cms.PSet(
            dX = cms.double(0.2),
            dY = cms.double(0.2),
        ),
    ),

    TOBHalfBarrels = cms.PSet(
        DetUnits = cms.PSet(
            dX = cms.double(0.2),
            dY = cms.double(0.2),
        ),
    ),

    TPEEndcaps = cms.PSet(
        DetUnits = cms.PSet(
            dX = cms.double(0.2),
            dY = cms.double(0.2),
        ),
    ),

    TIDEndcaps = cms.PSet(
        DetUnits = cms.PSet(
            dX = cms.double(0.2),
            dY = cms.double(0.2),
        ),
    ),

    TECEndcaps = cms.PSet(
        DetUnits = cms.PSet(
            dX = cms.double(0.2),
            dY = cms.double(0.2),
        ),
    ),
)

MisalignmentScenario_20muLocalXY = cms.PSet(
    MisalignmentScenarioSettings,
    scale = cms.double(0.01), # shifts in 100um

    TPBHalfBarrels = cms.PSet(
        DetUnits = cms.PSet(
            dXlocal = cms.double(0.2),
            dYlocal = cms.double(0.2),
        ),
    ),

    TIBHalfBarrels = cms.PSet(
        DetUnits = cms.PSet(
            dXlocal = cms.double(0.2),
            dYlocal = cms.double(0.2),
        ),
    ),

    TOBHalfBarrels = cms.PSet(
        DetUnits = cms.PSet(
            dXlocal = cms.double(0.2),
            dYlocal = cms.double(0.2),
        ),
    ),

    TPEEndcaps = cms.PSet(
        DetUnits = cms.PSet(
            dXlocal = cms.double(0.2),
            dYlocal = cms.double(0.2),
        ),
    ),

    TIDEndcaps = cms.PSet(
        DetUnits = cms.PSet(
            dXlocal = cms.double(0.2),
            dYlocal = cms.double(0.2),
        ),
    ),

    TECEndcaps = cms.PSet(
        DetUnits = cms.PSet(
            dXlocal = cms.double(0.2),
            dYlocal = cms.double(0.2),
        ),
    ),
)

MisalignmentScenario_20muGlobalXYZ = cms.PSet(
    MisalignmentScenarioSettings,
    scale = cms.double(0.01), # shifts in 100um

    TPBHalfBarrels = cms.PSet(
        DetUnits = cms.PSet(
            dX = cms.double(0.2),
            dY = cms.double(0.2),
            dZ = cms.double(0.2),
        ),
    ),

    TIBHalfBarrels = cms.PSet(
        DetUnits = cms.PSet(
            dX = cms.double(0.2),
            dY = cms.double(0.2),
            dZ = cms.double(0.2),
        ),
    ),

    TOBHalfBarrels = cms.PSet(
        DetUnits = cms.PSet(
            dX = cms.double(0.2),
            dY = cms.double(0.2),
            dZ = cms.double(0.2),
        ),
    ),

    TPEEndcaps = cms.PSet(
        DetUnits = cms.PSet(
            dX = cms.double(0.2),
            dY = cms.double(0.2),
            dZ = cms.double(0.2),
        ),
    ),

    TIDEndcaps = cms.PSet(
        DetUnits = cms.PSet(
            dX = cms.double(0.2),
            dY = cms.double(0.2),
            dZ = cms.double(0.2),
        ),
    ),

    TECEndcaps = cms.PSet(
        DetUnits = cms.PSet(
            dX = cms.double(0.2),
            dY = cms.double(0.2),
            dZ = cms.double(0.2),
        ),
    ),
)

MisalignmentScenario_20muLocalXYZ = cms.PSet(
    MisalignmentScenarioSettings,
    scale = cms.double(0.01), # shifts in 100um

    TPBHalfBarrels = cms.PSet(
        DetUnits = cms.PSet(
            dXlocal = cms.double(0.2),
            dYlocal = cms.double(0.2),
            dZlocal = cms.double(0.2),
        ),
    ),

    TIBHalfBarrels = cms.PSet(
        DetUnits = cms.PSet(
            dXlocal = cms.double(0.2),
            dYlocal = cms.double(0.2),
            dZlocal = cms.double(0.2),
        ),
    ),

    TOBHalfBarrels = cms.PSet(
        DetUnits = cms.PSet(
            dXlocal = cms.double(0.2),
            dYlocal = cms.double(0.2),
            dZlocal = cms.double(0.2),
        ),
    ),

    TPEEndcaps = cms.PSet(
        DetUnits = cms.PSet(
            dXlocal = cms.double(0.2),
            dYlocal = cms.double(0.2),
            dZlocal = cms.double(0.2),
        ),
    ),

    TIDEndcaps = cms.PSet(
        DetUnits = cms.PSet(
            dXlocal = cms.double(0.2),
            dYlocal = cms.double(0.2),
            dZlocal = cms.double(0.2),
        ),
    ),

    TECEndcaps = cms.PSet(
        DetUnits = cms.PSet(
            dXlocal = cms.double(0.2),
            dYlocal = cms.double(0.2),
            dZlocal = cms.double(0.2),
        ),
    ),
)

MisalignmentScenario_BPIX20muGlobalX = cms.PSet(
    MisalignmentScenarioSettings,
    scale = cms.double(0.01), # shifts in 100um

    TPBHalfBarrels = cms.PSet(
        DetUnits = cms.PSet(
            dX = cms.double(0.2),
        ),
    ),
)

MisalignmentScenario_BPIX20muGlobalY = cms.PSet(
    MisalignmentScenarioSettings,
    scale = cms.double(0.01), # shifts in 100um

    TPBHalfBarrels = cms.PSet(
        DetUnits = cms.PSet(
            dY = cms.double(0.2),
        ),
    ),
)

MisalignmentScenario_BPIX20muGlobalZ = cms.PSet(
    MisalignmentScenarioSettings,
    scale = cms.double(0.01), # shifts in 100um

    TPBHalfBarrels = cms.PSet(
        DetUnits = cms.PSet(
            dZ = cms.double(0.2),
        ),
    ),
)

MisalignmentScenario_BPIX20muLocalX = cms.PSet(
    MisalignmentScenarioSettings,
    scale = cms.double(0.01), # shifts in 100um

    TPBHalfBarrels = cms.PSet(
        DetUnits = cms.PSet(
            dXlocal = cms.double(0.2),
        ),
    ),
)

MisalignmentScenario_BPIX20muLocalY = cms.PSet(
    MisalignmentScenarioSettings,
    scale = cms.double(0.01), # shifts in 100um

    TPBHalfBarrels = cms.PSet(
        DetUnits = cms.PSet(
            dYlocal = cms.double(0.2),
        ),
    ),
)

MisalignmentScenario_BPIX20muLocalZ = cms.PSet(
    MisalignmentScenarioSettings,
    scale = cms.double(0.01), # shifts in 100um

    TPBHalfBarrels = cms.PSet(
        DetUnits = cms.PSet(
            dZlocal = cms.double(0.2),
        ),
    ),
)

MisalignmentScenario_NonMisalignedBPIX = cms.PSet(
    MisalignmentScenarioSettings,
    scale = cms.double(0.01), # shifts in 100um

    TPBHalfBarrels = cms.PSet(
        DetUnits = cms.PSet(
            #~ dYLocal = cms.double(0.1),
        ),
    ),

    TIBHalfBarrels = cms.PSet(
        DetUnits = cms.PSet(
            dXlocal = cms.double(0.25),
            dYlocal = cms.double(0.25),
            dZlocal = cms.double(0.25),
        ),
    ),

    TOBHalfBarrels = cms.PSet(
        DetUnits = cms.PSet(
            dXlocal = cms.double(0.25),
            dYlocal = cms.double(0.25),
            dZlocal = cms.double(0.25),
        ),
    ),

    TPEEndcaps = cms.PSet(
        DetUnits = cms.PSet(
            dXlocal = cms.double(0.25),
            dYlocal = cms.double(0.25),
            dZlocal = cms.double(0.25),
        ),
    ),

    TIDEndcaps = cms.PSet(
        DetUnits = cms.PSet(
            dXlocal = cms.double(0.25),
            dYlocal = cms.double(0.25),
            dZlocal = cms.double(0.25),
        ),
    ),

    TECEndcaps = cms.PSet(
        DetUnits = cms.PSet(
            dXlocal = cms.double(0.25),
            dYlocal = cms.double(0.25),
            dZlocal = cms.double(0.25),
        ),
    ),
)

MisalignmentScenarioDifferentSubdetectors = cms.PSet(
    MisalignmentScenarioSettings,
    scale = cms.double(0.01), # shifts in 100um

    TPBHalfBarrels = cms.PSet(
        DetUnits = cms.PSet(
            #~ dYLocal = cms.double(0.1),
        ),
    ),

    TIBHalfBarrels = cms.PSet(
        DetUnits = cms.PSet(
            dYlocal = cms.double(0.5),
            dZlocal = cms.double(0.5),
        ),
    ),

    TOBHalfBarrels = cms.PSet(
        DetUnits = cms.PSet(
            dXlocal = cms.double(0.05),
        ),
    ),

    TPEEndcaps = cms.PSet(
        DetUnits = cms.PSet(
            dZlocal = cms.double(0.4),
        ),
    ),

    TIDEndcaps = cms.PSet(
        DetUnits = cms.PSet(
            dXlocal = cms.double(0.1),
            dYlocal = cms.double(0.5),
            dZlocal = cms.double(0.5),
        ),
    ),

    TECEndcaps = cms.PSet(
        DetUnits = cms.PSet(
            dXlocal = cms.double(0.3),
        ),
    ),
)

MisalignmentScenarioDifferentSubdetectorsLarge = cms.PSet(
    MisalignmentScenarioSettings,
    scale = cms.double(0.01), # shifts in 100um

    TPBHalfBarrels = cms.PSet(
        DetUnits = cms.PSet(
            #~ dYLocal = cms.double(0.1),
        ),
    ),

    TIBHalfBarrels = cms.PSet(
        DetUnits = cms.PSet(
            dYlocal = cms.double(0.5),
            dZlocal = cms.double(0.5),
        ),
    ),

    TOBHalfBarrels = cms.PSet(
        DetUnits = cms.PSet(
            dXlocal = cms.double(0.25),
        ),
    ),

    TPEEndcaps = cms.PSet(
        DetUnits = cms.PSet(
            dZlocal = cms.double(0.4),
        ),
    ),

    TIDEndcaps = cms.PSet(
        DetUnits = cms.PSet(
            dXlocal = cms.double(0.15),
            dYlocal = cms.double(0.5),
            dZlocal = cms.double(0.5),
        ),
    ),

    TECEndcaps = cms.PSet(
        DetUnits = cms.PSet(
            dXlocal = cms.double(0.3),
        ),
    ),
)

MisalignmentScenarioDifferentSubdetectorsLocal = cms.PSet(
    MisalignmentScenarioSettings,
    #~ scale = cms.double(0.01), # shifts in 100um

    TPBHalfBarrels = cms.PSet(
        DetUnits = cms.PSet(
            dYlocal = cms.double(0.001),
        ),
    ),

    TIBHalfBarrels = cms.PSet(
        DetUnits = cms.PSet(
            dZlocal = cms.double(0.002),
        ),
    ),

    TOBHalfBarrels = cms.PSet(
        DetUnits = cms.PSet(
            phiYlocal = cms.double(0.01),
        ),
    ),

    TPEEndcaps = cms.PSet(
        DetUnits = cms.PSet(
            dXlocal = cms.double(0.002),
        ),
    ),

    TIDEndcaps = cms.PSet(
        DetUnits = cms.PSet(
            phiXlocal = cms.double(0.02),
        ),
    ),

    TECEndcaps = cms.PSet(
        DetUnits = cms.PSet(
            phiZlocal = cms.double(0.01),
        ),
    ),
)

MisalignmentScenario10Mu = cms.PSet(
    MisalignmentScenarioSettings,
    scale = cms.double(0.01), # shifts in 100um

    TPBHalfBarrels = cms.PSet(
        DetUnits = cms.PSet(
            dYlocal = cms.double(0.1),
            dXlocal = cms.double(0.1),
        ),
    ),

    TIBHalfBarrels = cms.PSet(
        DetUnits = cms.PSet(
            dYlocal = cms.double(0.1),
            dXlocal = cms.double(0.1),
        ),
    ),

    TOBHalfBarrels = cms.PSet(
        DetUnits = cms.PSet(
            dYlocal = cms.double(0.1),
            dXlocal = cms.double(0.1),
        ),
    ),

    TPEEndcaps = cms.PSet(
        DetUnits = cms.PSet(
            dYlocal = cms.double(0.1),
            dXlocal = cms.double(0.1),
        ),
    ),

    TIDEndcaps = cms.PSet(
        DetUnits = cms.PSet(
            dYlocal = cms.double(0.1),
            dXlocal = cms.double(0.1),
        ),
    ),

    TECEndcaps = cms.PSet(
        DetUnits = cms.PSet(
            dYlocal = cms.double(0.1),
            dXlocal = cms.double(0.1),
        ),
    ),
)

MisalignmentScenario100Mu = cms.PSet(
    MisalignmentScenarioSettings,
    scale = cms.double(0.01), # shifts in 100um

    TPBHalfBarrels = cms.PSet(
        DetUnits = cms.PSet(
            dZlocal = cms.double(1),
            dYlocal = cms.double(1),
            dXlocal = cms.double(1),
        ),
    ),

    TIBHalfBarrels = cms.PSet(
        DetUnits = cms.PSet(
            dZlocal = cms.double(1),
            dYlocal = cms.double(1),
            dXlocal = cms.double(1),
        ),
    ),

    TOBHalfBarrels = cms.PSet(
        DetUnits = cms.PSet(
            dZlocal = cms.double(1),
            dYlocal = cms.double(1),
            dXlocal = cms.double(1),
        ),
    ),

    TPEEndcaps = cms.PSet(
        DetUnits = cms.PSet(
            dZlocal = cms.double(1),
            dYlocal = cms.double(1),
            dXlocal = cms.double(1),
        ),
    ),

    TIDEndcaps = cms.PSet(
        DetUnits = cms.PSet(
            dZlocal = cms.double(1),
            dYlocal = cms.double(1),
            dXlocal = cms.double(1),
        ),
    ),

    TECEndcaps = cms.PSet(
        DetUnits = cms.PSet(
            dZlocal = cms.double(1),
            dYlocal = cms.double(1),
            dXlocal = cms.double(1),
        ),
    ),
)


MisalignmentScenario200Mu = cms.PSet(
    MisalignmentScenarioSettings,
    scale = cms.double(0.01),# shifts in 100mum

    TPBHalfBarrels = cms.PSet(
        DetUnits = cms.PSet(
            dZlocal = cms.double(2),
            dYlocal = cms.double(2),
            dXlocal = cms.double(2),
        ),
    ),

    TIBHalfBarrels = cms.PSet(
        DetUnits = cms.PSet(
            dZlocal = cms.double(2),
            dYlocal = cms.double(2),
            dXlocal = cms.double(2),
        ),
    ),

    TOBHalfBarrels = cms.PSet(
        DetUnits = cms.PSet(
            dZlocal = cms.double(2),
            dYlocal = cms.double(2),
            dXlocal = cms.double(2),
        ),
    ),

    TPEEndcaps = cms.PSet(
        DetUnits = cms.PSet(
            dZlocal = cms.double(2),
            dYlocal = cms.double(2),
            dXlocal = cms.double(2),
        ),
    ),

    TIDEndcaps = cms.PSet(
        DetUnits = cms.PSet(
            dZlocal = cms.double(2),
            dYlocal = cms.double(2),
            dXlocal = cms.double(2),
        ),
    ),

    TECEndcaps = cms.PSet(
        DetUnits = cms.PSet(
            dZlocal = cms.double(2),
            dYlocal = cms.double(2),
            dXlocal = cms.double(2),
        ),
    ),
)


MisalignmentScenario300Mu = cms.PSet(
    MisalignmentScenarioSettings,
    scale = cms.double(0.01),# shifts in 100mum

    TPBHalfBarrels = cms.PSet(
        DetUnits = cms.PSet(
            dZlocal = cms.double(3),
            dYlocal = cms.double(3),
            dXlocal = cms.double(3),
        ),
    ),

    TIBHalfBarrels = cms.PSet(
        DetUnits = cms.PSet(
            dZlocal = cms.double(3),
            dYlocal = cms.double(3),
            dXlocal = cms.double(3),
        ),
    ),

    TOBHalfBarrels = cms.PSet(
        DetUnits = cms.PSet(
            dZlocal = cms.double(3),
            dYlocal = cms.double(3),
            dXlocal = cms.double(3),
        ),
    ),

    TPEEndcaps = cms.PSet(
        DetUnits = cms.PSet(
            dZlocal = cms.double(3),
            dYlocal = cms.double(3),
            dXlocal = cms.double(3),
        ),
    ),

    TIDEndcaps = cms.PSet(
        DetUnits = cms.PSet(
            dZlocal = cms.double(3),
            dYlocal = cms.double(3),
            dXlocal = cms.double(3),
        ),
    ),

    TECEndcaps = cms.PSet(
        DetUnits = cms.PSet(
            dZlocal = cms.double(3),
            dYlocal = cms.double(3),
            dXlocal = cms.double(3),
        ),
    ),
)



MisalignmentScenarioBPIX100Mu = cms.PSet(
    MisalignmentScenarioSettings,
    scale = cms.double(0.01),# shifts in 100mum

    TPBHalfBarrels = cms.PSet(
        DetUnits = cms.PSet(
            dZlocal = cms.double(1),
            dYlocal = cms.double(1),
            dXlocal = cms.double(1),
        ),
    ),
)


MisalignedTPB = cms.PSet(
    MisalignmentScenarioSettings,
    scale = cms.double(0.01), # shifts in 100um

    TPBHalfBarrels = cms.PSet(
        DetUnits = cms.PSet(
            dZlocal = cms.double(1),
            dYlocal = cms.double(1),
            dXlocal = cms.double(1),
        )
    )
)

MisalignedTPE = cms.PSet(
    MisalignmentScenarioSettings,
    scale = cms.double(0.01), # shifts in 100um

    TPEEndcaps = cms.PSet(
        DetUnits = cms.PSet(
            dZlocal = cms.double(1),
            dYlocal = cms.double(1),
            dXlocal = cms.double(1),
        )
    )
)

MisalignedTIB = cms.PSet(
    MisalignmentScenarioSettings,
    scale = cms.double(0.01), # shifts in 100um

    TIBHalfBarrels = cms.PSet(
        DetUnits = cms.PSet(
            dZlocal = cms.double(1),
            dYlocal = cms.double(1),
            dXlocal = cms.double(1),
        )
    )
)

MisalignedTOB = cms.PSet(
    MisalignmentScenarioSettings,
    scale = cms.double(0.01), # shifts in 100um

    TOBHalfBarrels = cms.PSet(
        DetUnits = cms.PSet(
            dZlocal = cms.double(1),
            dYlocal = cms.double(1),
            dXlocal = cms.double(1),
        )
    )
)

MisalignedTID = cms.PSet(
    MisalignmentScenarioSettings,
    scale = cms.double(0.01), # shifts in 100um

    TIDEndcaps = cms.PSet(
        DetUnits = cms.PSet(
            dZlocal = cms.double(1),
            dYlocal = cms.double(1),
            dXlocal = cms.double(1),
        )
    )
)

MisalignedTEC = cms.PSet(
    MisalignmentScenarioSettings,
    scale = cms.double(0.01), # shifts in 100um

    TECEndcaps = cms.PSet(
        DetUnits = cms.PSet(
            dZlocal = cms.double(1),
            dYlocal = cms.double(1),
            dXlocal = cms.double(1),
        )
    )
)


MisalignmentAPEScenarioBase = cms.PSet(
    # Sigma in mum
    #   BPIX:   20
    #   BPIX-y: 10
    #   FPIX:   10
    #   FPIX-y: 20
    #   TEC:    20
    #   TIB:    10
    #   TID:    10
    #   TOB:    10 
    MisalignmentScenarioSettings,
    scale = cms.double(0.0001), # shifts in 1um

    TPBHalfBarrels = cms.PSet(
        DetUnits = cms.PSet(
            dZlocal = cms.double(20),
            dYlocal = cms.double(10),
            dXlocal = cms.double(20),
        ),
    ),

    TIBHalfBarrels = cms.PSet(
        DetUnits = cms.PSet(
            dZlocal = cms.double(10),
            dYlocal = cms.double(10),
            dXlocal = cms.double(10),
        ),
    ),

    TOBHalfBarrels = cms.PSet(
        DetUnits = cms.PSet(
            dZlocal = cms.double(10),
            dYlocal = cms.double(10),
            dXlocal = cms.double(10),
        ),
    ),

    TPEEndcaps = cms.PSet(
        DetUnits = cms.PSet(
            dZlocal = cms.double(20),
            dYlocal = cms.double(20),
            dXlocal = cms.double(10),
        ),
    ),

    TIDEndcaps = cms.PSet(
        DetUnits = cms.PSet(
            dZlocal = cms.double(10),
            dYlocal = cms.double(10),
            dXlocal = cms.double(10),
        ),
    ),

    TECEndcaps = cms.PSet(
        DetUnits = cms.PSet(
            dZlocal = cms.double(20),
            dYlocal = cms.double(20),
            dXlocal = cms.double(20),
        ),
    ),
)
