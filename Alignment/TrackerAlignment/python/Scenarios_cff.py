import FWCore.ParameterSet.Config as cms
import copy # for deepcopy used below

#
# This file contains all scenarios as blocks
# A block can be included in a config file as:
#   using <block label>
# in any place where a PSet could be used.
#
# See corresponding .cff files for examples.
#
# Note: following scenarios updated to new hierarchy
# with CMSSW_1_7_0 (numbers only copied, scenarios were
# not revised): 10/100/1000 pb-1, TrackerSurveyOnly,
# TrackerSurveyLAS
#
# Hierarchylevels not used in this update:
# TPBHalfBarrel
# TPBBarrel
# TPEEndcap
# TIBSurface
# TIBHalfShell
# TIBHalfBarrel
# TIBBarrel
# TIDSide
# TOBBarrel
# TECRing
# TECEndcap
# -----------------------------------------------------------------------
# General settings common to all scenarios
MisalignmentScenarioSettings = cms.PSet(
    setRotations = cms.bool(True),
    setTranslations = cms.bool(True),
    seed = cms.int32(1234567),
    distribution = cms.string('gaussian'),
    setError = cms.bool(True)
)
# -----------------------------------------------------------------------
#  "Misalignment" scenario without misalignment...
NoMovementsScenario = cms.PSet(
    MisalignmentScenarioSettings
)
# -----------------------------------------------------------------------
# Example scenario (dummy movements)
TrackerExampleScenario = cms.PSet(
      MisalignmentScenarioSettings,
      TOBHalfBarrel1 = cms.PSet(
        TOBLayers = cms.PSet(
            dX = cms.double(0.1)
            ),
        TOBLayer1 = cms.PSet(
            dX = cms.double(0.2)
            )
        ),
      TOBHalfBarrels = cms.PSet(
        TOBLayer1 = cms.PSet(
            phiX = cms.double(0.03)
            ),
        dX = cms.double(0.2)
        )
    )
# -----------------------------------------------------------------------
# 10 pb-1 misalignment scenario
# See CMS IN 2007/036
#
# first helper blocks for TEC:
TECRings10pb_1D = cms.PSet(
    DetUnits = cms.PSet(
        dZlocal = cms.double(0.0022),
        phiXlocal = cms.double(5e-05),
        dYlocal = cms.double(0.0022),
        phiZlocal = cms.double(5e-05),
        dXlocal = cms.double(0.0022),
        phiYlocal = cms.double(5e-05)
    )
)
TECRings10pb_2D = cms.PSet(
    Dets = cms.PSet(
        dZlocal = cms.double(0.0022),
        phiXlocal = cms.double(5e-05),
        dYlocal = cms.double(0.0022),
        phiZlocal = cms.double(5e-05),
        dXlocal = cms.double(0.0022),
        phiYlocal = cms.double(5e-05)
    )
)
# now real scenario:
Tracker10pbScenario = cms.PSet(
    MisalignmentScenarioSettings,
    TIBHalfBarrels = cms.PSet(
        TIBLayers = cms.PSet(
            dZlocal = cms.double(0.01),
            phiXlocal = cms.double(6.5e-05),
            dYlocal = cms.double(0.01),
            phiZlocal = cms.double(6.5e-05),
            dXlocal = cms.double(0.01),
            phiYlocal = cms.double(6.5e-05)
        ),
        scale = cms.double(1.0),
        TIBLayer3_4 = cms.PSet(
            DetUnits = cms.PSet(
                dZlocal = cms.double(0.018),
                phiXlocal = cms.double(0.000412),
                dYlocal = cms.double(0.018),
                phiZlocal = cms.double(0.000412),
                dXlocal = cms.double(0.018),
                phiYlocal = cms.double(0.000412)
            )
        ),
        scaleError = cms.double(1.0),
        TIBLayer1_2 = cms.PSet(
            Dets = cms.PSet(
                dZlocal = cms.double(0.018),
                phiXlocal = cms.double(0.000412),
                dYlocal = cms.double(0.018),
                phiZlocal = cms.double(0.000412),
                dXlocal = cms.double(0.018),
                phiYlocal = cms.double(0.000412)
            )
        ),
        distribution = cms.string('gaussian'),
        TIBStrings = cms.PSet(
            dZlocal = cms.double(0.01),
            phiXlocal = cms.double(6.5e-05),
            dYlocal = cms.double(0.01),
            phiZlocal = cms.double(6.5e-05),
            dXlocal = cms.double(0.01),
            phiYlocal = cms.double(6.5e-05)
        )
    ),
    TPBHalfBarrels = cms.PSet(
        scale = cms.double(1.0),
        DetUnits = cms.PSet(
            dZlocal = cms.double(0.006),
            phiXlocal = cms.double(0.00027),
            dYlocal = cms.double(0.006),
            phiZlocal = cms.double(0.00027),
            distribution = cms.string('gaussian'),
            dXlocal = cms.double(0.006),
            phiYlocal = cms.double(0.00027)
        ),
        TPBLayers = cms.PSet(
            phiXlocal = cms.double(7e-06),
            phiZlocal = cms.double(7e-06),
            dZ = cms.double(0.001),
            dX = cms.double(0.001),
            dY = cms.double(0.001),
            distribution = cms.string('gaussian'),
            phiYlocal = cms.double(7e-06)
        ),
        scaleError = cms.double(1.0),
        TPBLadders = cms.PSet(
            dZlocal = cms.double(0.001),
            phiXlocal = cms.double(7e-06),
            dYlocal = cms.double(0.001),
            phiZlocal = cms.double(7e-06),
            distribution = cms.string('gaussian'),
            dXlocal = cms.double(0.001),
            phiYlocal = cms.double(7e-06)
        )
    ),
    TOBHalfBarrels = cms.PSet(
        scale = cms.double(1.0),
        TOBLayer3_4_5_6 = cms.PSet(
            DetUnits = cms.PSet(
                dZlocal = cms.double(0.0032),
                phiXlocal = cms.double(7.5e-05),
                dYlocal = cms.double(0.0032),
                phiZlocal = cms.double(7.5e-05),
                dXlocal = cms.double(0.0032),
                phiYlocal = cms.double(7.5e-05)
            )
        ),
        scaleError = cms.double(1.0),
        TOBLayers = cms.PSet(
            dXlocal = cms.double(0.0),
            dZlocal = cms.double(0.0),
            dYlocal = cms.double(0.0)
        ),
        TOBLayer1_2 = cms.PSet(
            Dets = cms.PSet(
                dZlocal = cms.double(0.0032),
                phiXlocal = cms.double(7.5e-05),
                dYlocal = cms.double(0.0032),
                phiZlocal = cms.double(7.5e-05),
                dXlocal = cms.double(0.0032),
                phiYlocal = cms.double(7.5e-05)
            )
        ),
        TOBRods = cms.PSet(
            dZlocal = cms.double(0.01),
            phiXlocal = cms.double(4e-05),
            dYlocal = cms.double(0.01),
            phiZlocal = cms.double(4e-05),
            dXlocal = cms.double(0.01),
            phiYlocal = cms.double(4e-05)
        ),
        TOBHalfBarrels = cms.PSet(
            dZlocal = cms.double(0.01),
            phiXlocal = cms.double(1e-05),
            dYlocal = cms.double(0.006),
            phiZlocal = cms.double(1e-05),
            dXlocal = cms.double(0.006),
            phiYlocal = cms.double(1e-05)
        ),
        distribution = cms.string('gaussian')
    ),
    TECEndcaps = cms.PSet(
        scale = cms.double(1.0),
        TECDisk9 = cms.PSet(
            TECRing2 = cms.PSet(
                TECRings10pb_2D
            ),
            TECRing1_3_4 = cms.PSet(
                TECRings10pb_1D
            )
        ),
        scaleError = cms.double(1.0),
        TECPetals = cms.PSet(
            dZlocal = cms.double(0.007),
            phiXlocal = cms.double(3e-05),
            dYlocal = cms.double(0.007),
            phiZlocal = cms.double(3e-05),
            dXlocal = cms.double(0.007),
            phiYlocal = cms.double(3e-05)
        ),
        TECDisks = cms.PSet(
            dZlocal = cms.double(0.015),
            phiXlocal = cms.double(1.5e-05),
            dYlocal = cms.double(0.006),
            phiZlocal = cms.double(1.5e-05),
            dXlocal = cms.double(0.006),
            phiYlocal = cms.double(1.5e-05)
        ),
        TECDisk4_5_6 = cms.PSet(
            TECRing2_3_5_6 = cms.PSet(
                TECRings10pb_1D
            ),
            TECRing1_4 = cms.PSet(
                TECRings10pb_2D
            )
        ),
        TECDisk7_8 = cms.PSet(
            TECRing3 = cms.PSet(
                TECRings10pb_2D
            ),
            TECRing1_2_4_5 = cms.PSet(
                TECRings10pb_1D
            )
        ),
        distribution = cms.string('gaussian'),
        TECDisk1_2_3 = cms.PSet(
            TECRing3_4_6_7 = cms.PSet(
                TECRings10pb_1D
            ),
            TECRing1_2_5 = cms.PSet(
                TECRings10pb_2D
            )
        )
    ),
    TPEEndcaps = cms.PSet(
        TPEPanels = cms.PSet(
            dZlocal = cms.double(0.001),
            phiXlocal = cms.double(0.0002),
            dYlocal = cms.double(0.001),
            phiZlocal = cms.double(0.0002),
            dXlocal = cms.double(0.001),
            phiYlocal = cms.double(0.0002)
        ),
        scale = cms.double(1.0),
        scaleError = cms.double(1.0),
        TPEHalfDisks = cms.PSet(
            dZlocal = cms.double(0.001),
            phiXlocal = cms.double(1.5e-05),
            dYlocal = cms.double(0.001),
            phiZlocal = cms.double(1.5e-05),
            dXlocal = cms.double(0.001),
            phiYlocal = cms.double(1.5e-05)
        ),
        DetUnits = cms.PSet(
            dZlocal = cms.double(0.0005),
            phiXlocal = cms.double(0.0001),
            dYlocal = cms.double(0.0005),
            phiZlocal = cms.double(0.0001),
            dXlocal = cms.double(0.0005),
            phiYlocal = cms.double(0.0001)
        ),
        TPEBlades = cms.PSet(
            dZlocal = cms.double(0.001),
            phiXlocal = cms.double(1.5e-05),
            dYlocal = cms.double(0.001),
            phiZlocal = cms.double(1.5e-05),
            dXlocal = cms.double(0.001),
            phiYlocal = cms.double(1.5e-05)
        ),
        distribution = cms.string('gaussian')
    ),
    TIDEndcaps = cms.PSet(
        scale = cms.double(1.0),
        scaleError = cms.double(1.0),
        TIDRings = cms.PSet(
            dZlocal = cms.double(0.0185),
            phiXlocal = cms.double(0.00085),
            dYlocal = cms.double(0.0185),
            phiZlocal = cms.double(0.00085),
            dXlocal = cms.double(0.0185),
            phiYlocal = cms.double(0.00085)
        ),
        TIDRing1_2 = cms.PSet(
            Dets = cms.PSet(
                dZlocal = cms.double(0.0054),
                phiXlocal = cms.double(0.00025),
                dYlocal = cms.double(0.0054),
                phiZlocal = cms.double(0.00025),
                dXlocal = cms.double(0.0054),
                phiYlocal = cms.double(0.00025)
            )
        ),
        TIDDisks = cms.PSet(
            dZlocal = cms.double(0.025),
            phiXlocal = cms.double(0.00038),
            dYlocal = cms.double(0.025),
            phiZlocal = cms.double(0.00038),
            dXlocal = cms.double(0.025),
            phiYlocal = cms.double(0.00038)
        ),
        distribution = cms.string('gaussian'),
        TIDRing3 = cms.PSet(
            DetUnits = cms.PSet(
                dZlocal = cms.double(0.0054),
                phiXlocal = cms.double(0.00025),
                dYlocal = cms.double(0.0054),
                phiZlocal = cms.double(0.00025),
                dXlocal = cms.double(0.0054),
                phiYlocal = cms.double(0.00025)
            )
        )
    )
)
# 100 pb-1 misalignment scenario
# See CMS IN 2007/036
# first helper blocks for TEC:
TECRings100pb_1D = cms.PSet(
    DetUnits = cms.PSet(
        dZlocal = cms.double(0.0022),
        phiXlocal = cms.double(5e-05),
        dYlocal = cms.double(0.0022),
        phiZlocal = cms.double(5e-05),
        dXlocal = cms.double(0.0022),
        phiYlocal = cms.double(5e-05)
    )
)
TECRings100pb_2D = cms.PSet(
    Dets = cms.PSet(
        dZlocal = cms.double(0.0022),
        phiXlocal = cms.double(5e-05),
        dYlocal = cms.double(0.0022),
        phiZlocal = cms.double(5e-05),
        dXlocal = cms.double(0.0022),
        phiYlocal = cms.double(5e-05)
    )
)
# now real scenario:
Tracker100pbScenario = cms.PSet(
    MisalignmentScenarioSettings,
    TIBHalfBarrels = cms.PSet(
        TIBLayers = cms.PSet(
            dZlocal = cms.double(0.0015),
            phiXlocal = cms.double(1e-05),
            dYlocal = cms.double(0.0015),
            phiZlocal = cms.double(1e-05),
            dXlocal = cms.double(0.0015),
            phiYlocal = cms.double(1e-05)
        ),
        scale = cms.double(1.0),
        TIBLayer3_4 = cms.PSet(
            DetUnits = cms.PSet(
                dZlocal = cms.double(0.003),
                phiXlocal = cms.double(7e-05),
                dYlocal = cms.double(0.003),
                phiZlocal = cms.double(7e-05),
                dXlocal = cms.double(0.003),
                phiYlocal = cms.double(7e-05)
            )
        ),
        scaleError = cms.double(1.0),
        TIBLayer1_2 = cms.PSet(
            Dets = cms.PSet(
                dZlocal = cms.double(0.003),
                phiXlocal = cms.double(7e-05),
                dYlocal = cms.double(0.003),
                phiZlocal = cms.double(7e-05),
                dXlocal = cms.double(0.003),
                phiYlocal = cms.double(7e-05)
            )
        ),
        distribution = cms.string('gaussian'),
        TIBStrings = cms.PSet(
            dZlocal = cms.double(0.003),
            phiXlocal = cms.double(2e-05),
            dYlocal = cms.double(0.003),
            phiZlocal = cms.double(2e-05),
            dXlocal = cms.double(0.003),
            phiYlocal = cms.double(2e-05)
        )
    ),
    TPBHalfBarrels = cms.PSet(
        scale = cms.double(1.0),
        DetUnits = cms.PSet(
            dZlocal = cms.double(0.001),
            phiXlocal = cms.double(4.5e-05),
            dYlocal = cms.double(0.001),
            phiZlocal = cms.double(4.5e-05),
            distribution = cms.string('gaussian'),
            dXlocal = cms.double(0.001),
            phiYlocal = cms.double(4.5e-05)
        ),
        TPBLayers = cms.PSet(
            phiXlocal = cms.double(3e-06),
            phiZlocal = cms.double(3e-06),
            dZ = cms.double(0.0005),
            dX = cms.double(0.0005),
            dY = cms.double(0.0005),
            distribution = cms.string('gaussian'),
            phiYlocal = cms.double(3e-06)
        ),
        scaleError = cms.double(1.0),
        TPBLadders = cms.PSet(
            dZlocal = cms.double(0.001),
            phiXlocal = cms.double(7e-06),
            dYlocal = cms.double(0.001),
            phiZlocal = cms.double(7e-06),
            distribution = cms.string('gaussian'),
            dXlocal = cms.double(0.001),
            phiYlocal = cms.double(7e-06)
        )
    ),
    TOBHalfBarrels = cms.PSet(
        scale = cms.double(1.0),
        TOBLayer3_4_5_6 = cms.PSet(
            DetUnits = cms.PSet(
                dZlocal = cms.double(0.0032),
                phiXlocal = cms.double(7e-05),
                dYlocal = cms.double(0.0032),
                phiZlocal = cms.double(7e-05),
                dXlocal = cms.double(0.0032),
                phiYlocal = cms.double(7e-05)
            )
        ),
        scaleError = cms.double(1.0),
        TOBLayers = cms.PSet(
            dXlocal = cms.double(0.0),
            dZlocal = cms.double(0.0),
            dYlocal = cms.double(0.0)
        ),
        TOBLayer1_2 = cms.PSet(
            Dets = cms.PSet(
                dZlocal = cms.double(0.0032),
                phiXlocal = cms.double(7e-05),
                dYlocal = cms.double(0.0032),
                phiZlocal = cms.double(7e-05),
                dXlocal = cms.double(0.0032),
                phiYlocal = cms.double(7e-05)
            )
        ),
        TOBRods = cms.PSet(

            dZlocal = cms.double(0.004),
            phiXlocal = cms.double(1.5e-05),
            dYlocal = cms.double(0.004),
            phiZlocal = cms.double(1.5e-05),
            dXlocal = cms.double(0.004),
            phiYlocal = cms.double(1.5e-05)
        ),
        TOBHalfBarrels = cms.PSet(
            dZlocal = cms.double(0.002),
            phiXlocal = cms.double(5e-06),
            dYlocal = cms.double(0.002),
            phiZlocal = cms.double(5e-06),
            dXlocal = cms.double(0.002),
            phiYlocal = cms.double(5e-06)
        ),
        distribution = cms.string('gaussian')
    ),
    TECEndcaps = cms.PSet(
        scale = cms.double(1.0),
        TECDisk9 = cms.PSet(
            TECRing2 = cms.PSet(
                TECRings100pb_2D
            ),
            TECRing1_3_4 = cms.PSet(
                TECRings100pb_1D
            )
        ),
        scaleError = cms.double(1.0),
        TECPetals = cms.PSet(
            dZlocal = cms.double(0.0055),
            phiXlocal = cms.double(2e-05),
            dYlocal = cms.double(0.0055),
            phiZlocal = cms.double(2e-05),
            dXlocal = cms.double(0.0055),
            phiYlocal = cms.double(2e-05)
        ),
        TECDisks = cms.PSet(
            dZlocal = cms.double(0.003),
            phiXlocal = cms.double(5e-06),
            dYlocal = cms.double(0.003),
            phiZlocal = cms.double(5e-06),
            dXlocal = cms.double(0.003),
            phiYlocal = cms.double(5e-06)
        ),
        TECDisk4_5_6 = cms.PSet(
            TECRing2_3_5_6 = cms.PSet(
                TECRings100pb_1D
            ),
            TECRing1_4 = cms.PSet(
                TECRings100pb_2D
            )
        ),
        TECDisk7_8 = cms.PSet(
            TECRing3 = cms.PSet(
                TECRings100pb_2D
            ),
            TECRing1_2_4_5 = cms.PSet(
                TECRings100pb_1D
            )
        ),
        distribution = cms.string('gaussian'),
        TECDisk1_2_3 = cms.PSet(
            TECRing3_4_6_7 = cms.PSet(
                TECRings100pb_1D
            ),
            TECRing1_2_5 = cms.PSet(
                TECRings100pb_2D
            )
        )
    ),
    TPEEndcaps = cms.PSet(
        TPEPanels = cms.PSet(
            dZlocal = cms.double(0.001),
            phiXlocal = cms.double(2.2e-05),
            dYlocal = cms.double(0.001),
            phiZlocal = cms.double(2.2e-05),
            dXlocal = cms.double(0.001),
            phiYlocal = cms.double(2.2e-05)
        ),
        scale = cms.double(1.0),
        scaleError = cms.double(1.0),
        TPEHalfDisks = cms.PSet(
            dZlocal = cms.double(0.001),
            phiXlocal = cms.double(1.5e-05),
            dYlocal = cms.double(0.001),
            phiZlocal = cms.double(1.5e-05),
            dXlocal = cms.double(0.001),
            phiYlocal = cms.double(1.5e-05)
        ),
        DetUnits = cms.PSet(
            dZlocal = cms.double(0.0005),
            phiXlocal = cms.double(1.1e-05),
            dYlocal = cms.double(0.0005),
            phiZlocal = cms.double(1.1e-05),
            dXlocal = cms.double(0.0005),
            phiYlocal = cms.double(1.1e-05)
        ),
        TPEBlades = cms.PSet(
            dZlocal = cms.double(0.001),
            phiXlocal = cms.double(1.5e-05),
            dYlocal = cms.double(0.001),
            phiZlocal = cms.double(1.5e-05),
            dXlocal = cms.double(0.001),
            phiYlocal = cms.double(1.5e-05)
        ),
        distribution = cms.string('gaussian')
    ),
    TIDEndcaps = cms.PSet(
        scale = cms.double(1.0),
        scaleError = cms.double(1.0),
        TIDRing3 = cms.PSet(
            DetUnits = cms.PSet(
                dZlocal = cms.double(0.005),
                phiXlocal = cms.double(0.00023),
                dYlocal = cms.double(0.005),
                phiZlocal = cms.double(0.00023),
                dXlocal = cms.double(0.005),
                phiYlocal = cms.double(0.00023)
            )
        ),
        TIDRing1_2 = cms.PSet(
            Dets = cms.PSet(
                dZlocal = cms.double(0.005),
                phiXlocal = cms.double(0.00023),
                dYlocal = cms.double(0.005),
                phiZlocal = cms.double(0.00023),
                dXlocal = cms.double(0.005),
                phiYlocal = cms.double(0.00023)
            )
        ),
        TIDDisks = cms.PSet(
            dZlocal = cms.double(0.0025),
            phiXlocal = cms.double(4e-05),
            dYlocal = cms.double(0.0025),
            phiZlocal = cms.double(4e-05),
            dXlocal = cms.double(0.0025),
            phiYlocal = cms.double(4e-05)
        ),
        distribution = cms.string('gaussian'),
        TIDRings = cms.PSet(
            dZlocal = cms.double(0.005),
            phiXlocal = cms.double(0.00023),
            dYlocal = cms.double(0.005),
            phiZlocal = cms.double(0.00023),
            dXlocal = cms.double(0.005),
            phiYlocal = cms.double(0.00023)
        )
    )
)
# -----------------------------------------------------------------------
# 1000 pb-1 misalignment scenario
# See CMS IN 2007/036
# first helper blocks for TEC:
TECRings1000pb_1D = cms.PSet(
    DetUnits = cms.PSet(
        dZlocal = cms.double(0.0022),
        phiXlocal = cms.double(5e-05),
        dYlocal = cms.double(0.0022),
        phiZlocal = cms.double(5e-05),
        dXlocal = cms.double(0.0022),
        phiYlocal = cms.double(5e-05)
    )
)
TECRings1000pb_2D = cms.PSet(
    Dets = cms.PSet(
        dZlocal = cms.double(0.0022),
        phiXlocal = cms.double(5e-05),
        dYlocal = cms.double(0.0022),
        phiZlocal = cms.double(5e-05),
        dXlocal = cms.double(0.0022),
        phiYlocal = cms.double(5e-05)
    )
)
# now real scenario:
Tracker1000pbScenario = cms.PSet(
    MisalignmentScenarioSettings,
    TIBHalfBarrels = cms.PSet(
        TIBLayers = cms.PSet(
            dZlocal = cms.double(0.001),
            phiXlocal = cms.double(5e-06),
            dYlocal = cms.double(0.001),
            phiZlocal = cms.double(5e-06),
            dXlocal = cms.double(0.001),
            phiYlocal = cms.double(5e-06)
        ),
        scale = cms.double(1.0),
        TIBLayer3_4 = cms.PSet(
            DetUnits = cms.PSet(
                dZlocal = cms.double(0.001),
                phiXlocal = cms.double(2e-05),
                dYlocal = cms.double(0.001),
                phiZlocal = cms.double(2e-05),
                dXlocal = cms.double(0.001),
                phiYlocal = cms.double(2e-05)
            )
        ),
        scaleError = cms.double(1.0),
        TIBLayer1_2 = cms.PSet(
            Dets = cms.PSet(
                dZlocal = cms.double(0.001),
                phiXlocal = cms.double(2e-05),
                dYlocal = cms.double(0.001),
                phiZlocal = cms.double(2e-05),
                dXlocal = cms.double(0.001),
                phiYlocal = cms.double(2e-05)
            )
        ),
        distribution = cms.string('gaussian'),
        TIBStrings = cms.PSet(
            dZlocal = cms.double(0.001),
            phiXlocal = cms.double(5e-06),
            dYlocal = cms.double(0.001),
            phiZlocal = cms.double(5e-06),
            dXlocal = cms.double(0.001),
            phiYlocal = cms.double(5e-06)
        )
    ),
    TPBHalfBarrels = cms.PSet(
        scale = cms.double(1.0),
        DetUnits = cms.PSet(
            dZlocal = cms.double(0.0005),
            phiXlocal = cms.double(2.2e-05),
            dYlocal = cms.double(0.0005),
            phiZlocal = cms.double(2.2e-05),
            distribution = cms.string('gaussian'),
            dXlocal = cms.double(0.0005),
            phiYlocal = cms.double(2.2e-05)
        ),
        TPBLayers = cms.PSet(
            phiXlocal = cms.double(3e-06),
            phiZlocal = cms.double(3e-06),
            dZ = cms.double(0.0005),
            dX = cms.double(0.0005),
            dY = cms.double(0.0005),
            distribution = cms.string('gaussian'),
            phiYlocal = cms.double(3e-06)
        ),
        scaleError = cms.double(1.0),
        TPBLadders = cms.PSet(
            dZlocal = cms.double(0.0005),
            phiXlocal = cms.double(3e-06),
            dYlocal = cms.double(0.0005),
            phiZlocal = cms.double(3e-06),
            distribution = cms.string('gaussian'),
            dXlocal = cms.double(0.0005),
            phiYlocal = cms.double(3e-06)
        )
    ),
    TOBHalfBarrels = cms.PSet(
        scale = cms.double(1.0),
        TOBLayer3_4_5_6 = cms.PSet(
            DetUnits = cms.PSet(
                dZlocal = cms.double(0.0018),
                phiXlocal = cms.double(4e-05),
                dYlocal = cms.double(0.0018),
                phiZlocal = cms.double(4e-05),
                dXlocal = cms.double(0.0018),
                phiYlocal = cms.double(4e-05)
            )
        ),
        scaleError = cms.double(1.0),
        TOBLayers = cms.PSet(
            dXlocal = cms.double(0.0),
            dZlocal = cms.double(0.0),
            dYlocal = cms.double(0.0)
        ),
        TOBLayer1_2 = cms.PSet(
            Dets = cms.PSet(
                dZlocal = cms.double(0.0018),
                phiXlocal = cms.double(4e-05),
                dYlocal = cms.double(0.0018),
                phiZlocal = cms.double(4e-05),
                dXlocal = cms.double(0.0018),
                phiYlocal = cms.double(4e-05)
            )
        ),
        TOBRods = cms.PSet(
            dZlocal = cms.double(0.0018),
            phiXlocal = cms.double(7e-06),
            dYlocal = cms.double(0.0018),
            phiZlocal = cms.double(7e-06),
            dXlocal = cms.double(0.0018),
            phiYlocal = cms.double(7e-06)
        ),
        TOBHalfBarrels = cms.PSet(
            dZlocal = cms.double(0.001),
            phiXlocal = cms.double(2e-06),
            dYlocal = cms.double(0.001),
            phiZlocal = cms.double(2e-06),
            dXlocal = cms.double(0.001),
            phiYlocal = cms.double(2e-06)
        ),
        distribution = cms.string('gaussian')
    ),
    TECEndcaps = cms.PSet(
        scale = cms.double(1.0),
        TECDisk9 = cms.PSet(
            TECRing2 = cms.PSet(
                TECRings1000pb_2D
            ),
            TECRing1_3_4 = cms.PSet(
                TECRings1000pb_1D
            )
        ),
        scaleError = cms.double(1.0),
        TECPetals = cms.PSet(
            dZlocal = cms.double(0.004),
            phiXlocal = cms.double(1.5e-05),
            dYlocal = cms.double(0.004),
            phiZlocal = cms.double(1.5e-05),
            dXlocal = cms.double(0.004),
            phiYlocal = cms.double(1.5e-05)
        ),
        TECDisks = cms.PSet(
            dZlocal = cms.double(0.002),
            phiXlocal = cms.double(5e-06),
            dYlocal = cms.double(0.002),
            phiZlocal = cms.double(5e-06),
            dXlocal = cms.double(0.002),
            phiYlocal = cms.double(5e-06)
        ),
        TECDisk4_5_6 = cms.PSet(
            TECRing2_3_5_6 = cms.PSet(
                TECRings1000pb_1D
            ),
            TECRing1_4 = cms.PSet(
                TECRings1000pb_2D
            )
        ),
        TECDisk7_8 = cms.PSet(
            TECRing3 = cms.PSet(
                TECRings1000pb_2D
            ),
            TECRing1_2_4_5 = cms.PSet(
                TECRings1000pb_1D
            )
        ),
        distribution = cms.string('gaussian'),
        TECDisk1_2_3 = cms.PSet(
            TECRing3_4_6_7 = cms.PSet(
                TECRings1000pb_1D
            ),
            TECRing1_2_5 = cms.PSet(
                TECRings1000pb_2D
            )
        )
    ),
    TPEEndcaps = cms.PSet(
        TPEPanels = cms.PSet(
            dZlocal = cms.double(0.0005),
            phiXlocal = cms.double(1.1e-05),
            dYlocal = cms.double(0.0005),
            phiZlocal = cms.double(1.1e-05),
            dXlocal = cms.double(0.0005),
            phiYlocal = cms.double(1.1e-05)
        ),
        scale = cms.double(1.0),
        scaleError = cms.double(1.0),
        TPEHalfDisks = cms.PSet(
            dZlocal = cms.double(0.0005),
            phiXlocal = cms.double(7e-06),
            dYlocal = cms.double(0.0005),
            phiZlocal = cms.double(7e-06),
            dXlocal = cms.double(0.0005),
            phiYlocal = cms.double(7e-06)
        ),
        DetUnits = cms.PSet(
            dZlocal = cms.double(0.0005),
            phiXlocal = cms.double(1.1e-05),
            dYlocal = cms.double(0.0005),
            phiZlocal = cms.double(1.1e-05),
            dXlocal = cms.double(0.0005),
            phiYlocal = cms.double(1.1e-05)
        ),
        TPEBlades = cms.PSet(
            dZlocal = cms.double(0.0005),
            phiXlocal = cms.double(7e-06),
            dYlocal = cms.double(0.0005),
            phiZlocal = cms.double(7e-06),
            dXlocal = cms.double(0.0005),
            phiYlocal = cms.double(7e-06)
        ),
        distribution = cms.string('gaussian')
    ),
    TIDEndcaps = cms.PSet(
        scale = cms.double(1.0),
        scaleError = cms.double(1.0),
        TIDRing3 = cms.PSet(
            DetUnits = cms.PSet(
                dZlocal = cms.double(0.0025),
                phiXlocal = cms.double(0.00011),
                dYlocal = cms.double(0.0025),
                phiZlocal = cms.double(0.00011),
                dXlocal = cms.double(0.0025),
                phiYlocal = cms.double(0.00011)
            )
        ),
        TIDRing1_2 = cms.PSet(
            Dets = cms.PSet(
                dZlocal = cms.double(0.0025),
                phiXlocal = cms.double(0.00011),
                dYlocal = cms.double(0.0025),
                phiZlocal = cms.double(0.00011),
                dXlocal = cms.double(0.0025),
                phiYlocal = cms.double(0.00011)
            )
        ),
        TIDDisks = cms.PSet(
            dZlocal = cms.double(0.0012),
            phiXlocal = cms.double(2e-05),
            dYlocal = cms.double(0.0012),
            phiZlocal = cms.double(2e-05),
            dXlocal = cms.double(0.0012),
            phiYlocal = cms.double(2e-05)
        ),
        distribution = cms.string('gaussian'),
        TIDRings = cms.PSet(
            dZlocal = cms.double(0.0025),
            phiXlocal = cms.double(0.00011),
            dYlocal = cms.double(0.0025),
            phiZlocal = cms.double(0.00011),
            dXlocal = cms.double(0.0025),
            phiYlocal = cms.double(0.00011)
        )
    )
)
# -----------------------------------------------------------------------
# Survey&LAS only misalignment scenario
# See CMS IN 2007/036, table 6, "Updated initial uncertainties"
# first helper blocks for TEC:
TECRingsSurveyLASOnly_1D = cms.PSet(
    DetUnits = cms.PSet(
        dZlocal = cms.double(0.0022),
        phiXlocal = cms.double(5e-05),
        dYlocal = cms.double(0.0022),
        phiZlocal = cms.double(5e-05),
        dXlocal = cms.double(0.0022),
        phiYlocal = cms.double(5e-05)
    )
)
TECRingsSurveyLASOnly_2D = cms.PSet(
    Dets = cms.PSet(
        dZlocal = cms.double(0.0022),
        phiXlocal = cms.double(5e-05),
        dYlocal = cms.double(0.0022),
        phiZlocal = cms.double(5e-05),
        dXlocal = cms.double(0.0022),
        phiYlocal = cms.double(5e-05)
    )
)
# now real scenario:
TrackerSurveyLASOnlyScenario = cms.PSet(
    MisalignmentScenarioSettings,
    TIBHalfBarrels = cms.PSet(
        TIBLayers = cms.PSet(
            dZlocal = cms.double(0.075),
            phiXlocal = cms.double(0.000488),
            dYlocal = cms.double(0.075),
            phiZlocal = cms.double(0.000488),
            dXlocal = cms.double(0.075),
            phiYlocal = cms.double(0.000488)
        ),
        scale = cms.double(1.0),
        TIBLayer3_4 = cms.PSet(
            DetUnits = cms.PSet(
                dZlocal = cms.double(0.018),
                phiXlocal = cms.double(0.000412),
                dYlocal = cms.double(0.018),
                phiZlocal = cms.double(0.000412),
                dXlocal = cms.double(0.018),
                phiYlocal = cms.double(0.000412)
            )
        ),
        scaleError = cms.double(1.0),
        TIBLayer1_2 = cms.PSet(
            Dets = cms.PSet(
                dZlocal = cms.double(0.018),
                phiXlocal = cms.double(0.000412),
                dYlocal = cms.double(0.018),
                phiZlocal = cms.double(0.000412),
                dXlocal = cms.double(0.018),
                phiYlocal = cms.double(0.000412)
            )
        ),
        distribution = cms.string('gaussian'),
        TIBStrings = cms.PSet(
            dZlocal = cms.double(0.045),
            phiXlocal = cms.double(0.000293),
            dYlocal = cms.double(0.045),
            phiZlocal = cms.double(0.000293),
            dXlocal = cms.double(0.045),
            phiYlocal = cms.double(0.000293)
        )
    ),
    TPBHalfBarrels = cms.PSet(
        scale = cms.double(1.0),
        DetUnits = cms.PSet(
            dZlocal = cms.double(0.006),
            phiXlocal = cms.double(0.00027),
            dYlocal = cms.double(0.006),
            phiZlocal = cms.double(0.00027),
            distribution = cms.string('gaussian'),
            dXlocal = cms.double(0.006),
            phiYlocal = cms.double(0.00027)
        ),
        TPBLayers = cms.PSet(
            phiXlocal = cms.double(1e-05),
            phiZlocal = cms.double(1e-05),
            dZ = cms.double(0.0337),
            dX = cms.double(0.0225),
            dY = cms.double(0.0225),
            distribution = cms.string('gaussian'),
            phiYlocal = cms.double(1e-05)
        ),
        scaleError = cms.double(1.0),
        TPBLadders = cms.PSet(
            dZlocal = cms.double(0.005),
            phiXlocal = cms.double(2e-05),
            dYlocal = cms.double(0.005),
            phiZlocal = cms.double(2e-05),
            distribution = cms.string('flat'),
            dXlocal = cms.double(0.005),
            phiYlocal = cms.double(2e-05)
        )
    ),
    TOBHalfBarrels = cms.PSet(
        scale = cms.double(1.0),
        TOBLayer3_4_5_6 = cms.PSet(
            DetUnits = cms.PSet(
                dZlocal = cms.double(0.0032),
                phiXlocal = cms.double(7.5e-05),
                dYlocal = cms.double(0.0032),
                phiZlocal = cms.double(7.5e-05),
                dXlocal = cms.double(0.0032),
                phiYlocal = cms.double(7.5e-05)
            )
        ),
        scaleError = cms.double(1.0),
        TOBLayers = cms.PSet(
            dXlocal = cms.double(0.0),
            dZlocal = cms.double(0.0),
            dYlocal = cms.double(0.0)
        ),
        TOBLayer1_2 = cms.PSet(
            Dets = cms.PSet(
                dZlocal = cms.double(0.0032),
                phiXlocal = cms.double(7.5e-05),
                dYlocal = cms.double(0.0032),
                phiZlocal = cms.double(7.5e-05),
                dXlocal = cms.double(0.0032),
                phiYlocal = cms.double(7.5e-05)
            )
        ),
        TOBRods = cms.PSet(
            dZlocal = cms.double(0.01),
            phiXlocal = cms.double(4e-05),
            dYlocal = cms.double(0.01),
            phiZlocal = cms.double(4e-05),
            dXlocal = cms.double(0.01),
            phiYlocal = cms.double(4e-05)
        ),
        TOBHalfBarrels = cms.PSet(
            dZlocal = cms.double(0.05),
            phiXlocal = cms.double(1e-05),
            dYlocal = cms.double(0.006),
            phiZlocal = cms.double(1e-05),
            dXlocal = cms.double(0.006),
            phiYlocal = cms.double(1e-05)
        ),
        distribution = cms.string('gaussian')
    ),
    TECEndcaps = cms.PSet(
        scale = cms.double(1.0),
        TECDisk9 = cms.PSet(
            TECRing2 = cms.PSet(
                TECRingsSurveyLASOnly_2D
            ),
            TECRing1_3_4 = cms.PSet(
                TECRingsSurveyLASOnly_1D
            )
        ),
        scaleError = cms.double(1.0),
        TECPetals = cms.PSet(
            dZlocal = cms.double(0.007),
            phiXlocal = cms.double(3e-05),
            dYlocal = cms.double(0.007),
            phiZlocal = cms.double(3e-05),
            dXlocal = cms.double(0.007),
            phiYlocal = cms.double(3e-05)
        ),
        TECDisks = cms.PSet(
            dZlocal = cms.double(0.015),
            phiXlocal = cms.double(1.5e-05),
            dYlocal = cms.double(0.006),
            phiZlocal = cms.double(1.5e-05),
            dXlocal = cms.double(0.006),
            phiYlocal = cms.double(1.5e-05)
        ),
        TECDisk4_5_6 = cms.PSet(
            TECRing2_3_5_6 = cms.PSet(
                TECRingsSurveyLASOnly_1D
            ),
            TECRing1_4 = cms.PSet(
                TECRingsSurveyLASOnly_2D
            )
        ),
        TECDisk7_8 = cms.PSet(
            TECRing3 = cms.PSet(
                TECRingsSurveyLASOnly_2D
            ),
            TECRing1_2_4_5 = cms.PSet(
                TECRingsSurveyLASOnly_1D
            )
        ),
        distribution = cms.string('gaussian'),
        TECDisk1_2_3 = cms.PSet(
            TECRing3_4_6_7 = cms.PSet(
                TECRingsSurveyLASOnly_1D
            ),
            TECRing1_2_5 = cms.PSet(
                TECRingsSurveyLASOnly_2D
            )
        )
    ),
    TPEEndcaps = cms.PSet(
        TPEPanels = cms.PSet(
            dZlocal = cms.double(0.001),
            phiXlocal = cms.double(0.0002),
            dYlocal = cms.double(0.001),
            phiZlocal = cms.double(0.0002),
            dXlocal = cms.double(0.001),
            phiYlocal = cms.double(0.0002)
        ),
        scale = cms.double(1.0),
        scaleError = cms.double(1.0),
        TPEHalfDisks = cms.PSet(
            dZlocal = cms.double(0.005),
            phiXlocal = cms.double(0.001),
            dYlocal = cms.double(0.005),
            phiZlocal = cms.double(0.001),
            dXlocal = cms.double(0.005),
            phiYlocal = cms.double(0.001)
        ),
        TPEHalfCylinders = cms.PSet(
            dZlocal = cms.double(0.005),
            phiXlocal = cms.double(0.001),
            dYlocal = cms.double(0.005),
            phiZlocal = cms.double(0.001),
            dXlocal = cms.double(0.005),
            phiYlocal = cms.double(0.001)
        ),
        DetUnits = cms.PSet(
            dZlocal = cms.double(0.0005),
            phiXlocal = cms.double(0.0001),
            dYlocal = cms.double(0.0005),
            phiZlocal = cms.double(0.0001),
            dXlocal = cms.double(0.0005),
            phiYlocal = cms.double(0.0001)
        ),
        TPEBlades = cms.PSet(
            dZlocal = cms.double(0.001),
            phiXlocal = cms.double(0.0002),
            dYlocal = cms.double(0.001),
            phiZlocal = cms.double(0.0002),
            dXlocal = cms.double(0.001),
            phiYlocal = cms.double(0.0002)
        ),
        distribution = cms.string('gaussian')
    ),
    TIDEndcaps = cms.PSet(
        scale = cms.double(1.0),
        scaleError = cms.double(1.0),
        TIDEndcaps = cms.PSet(
            dZlocal = cms.double(0.045),
            phiXlocal = cms.double(0.000649),
            dYlocal = cms.double(0.045),
            phiZlocal = cms.double(0.000649),
            dXlocal = cms.double(0.045),
            phiYlocal = cms.double(0.000649)
        ),
        TIDRing3 = cms.PSet(
            DetUnits = cms.PSet(
                dZlocal = cms.double(0.0054),
                phiXlocal = cms.double(0.00025),
                dYlocal = cms.double(0.0054),
                phiZlocal = cms.double(0.00025),
                dXlocal = cms.double(0.0054),
                phiYlocal = cms.double(0.00025)
            )
        ),
        TIDRing1_2 = cms.PSet(
            Dets = cms.PSet(
                dZlocal = cms.double(0.0054),
                phiXlocal = cms.double(0.00025),
                dYlocal = cms.double(0.0054),
                phiZlocal = cms.double(0.00025),
                dXlocal = cms.double(0.0054),
                phiYlocal = cms.double(0.00025)
            )
        ),
        TIDDisks = cms.PSet(
            dZlocal = cms.double(0.035),
            phiXlocal = cms.double(0.000532),
            dYlocal = cms.double(0.035),
            phiZlocal = cms.double(0.000532),
            dXlocal = cms.double(0.035),
            phiYlocal = cms.double(0.000532)
        ),
        distribution = cms.string('gaussian'),
        TIDRings = cms.PSet(
            dZlocal = cms.double(0.0185),
            phiXlocal = cms.double(0.00085),
            dYlocal = cms.double(0.0185),
            phiZlocal = cms.double(0.00085),
            dXlocal = cms.double(0.0185),
            phiYlocal = cms.double(0.00085)
        )
    )
)
# -----------------------------------------------------------------------
# Survey&LAS&Cosmics scenario
# ----------------------------------------
# ----------------- NOTE -----------------
# Sufficient studies do not yet exist to
# provide a reliable version of this 
# scenario (Survey+LAS+Cosmics alignment).
#
# This scenario is not supposed to be used
# to make public(?) estimates of the 
# performance of the CMS.  
#
# This scenario contains lots of guesses,
# especially concerning the improvement
# one can reach by using Cosmics in
# track based alignment.
# The guess is that with Cosmics, one 
# can reach for largest barrel-like parts
# the average alignment accuracy of the 
# 10pb-1 and the SurveyLASOnly scenarios.
#
# The same applies also,but to a lesser
# extent, to the 10pb-1 scenario.
# ------------- NOTE ends ----------------
# ----------------------------------------
# first helper blocks for TEC:
TECRingsSurveyLASCosmics_1D = cms.PSet(
    DetUnits = cms.PSet(
        dZlocal = cms.double(0.0022),
        phiXlocal = cms.double(5e-05),
        dYlocal = cms.double(0.0022),
        phiZlocal = cms.double(5e-05),
        dXlocal = cms.double(0.0022),
        phiYlocal = cms.double(5e-05)
    )
)
TECRingsSurveyLASCosmics_2D = cms.PSet(
    Dets = cms.PSet(
        dZlocal = cms.double(0.0022),
        phiXlocal = cms.double(5e-05),
        dYlocal = cms.double(0.0022),
        phiZlocal = cms.double(5e-05),
        dXlocal = cms.double(0.0022),
        phiYlocal = cms.double(5e-05)
    )
)
# now real scenario:
TrackerSurveyLASCosmicsScenario = cms.PSet(
    MisalignmentScenarioSettings,
    TIBHalfBarrels = cms.PSet(
        TIBLayers = cms.PSet(
            dZlocal = cms.double(0.0425),
            phiXlocal = cms.double(0.000277),
            dYlocal = cms.double(0.0425),
            phiZlocal = cms.double(0.000277),
            dXlocal = cms.double(0.0425),
            phiYlocal = cms.double(0.000277)
        ),
        scale = cms.double(1.0),
        TIBLayer3_4 = cms.PSet(
            DetUnits = cms.PSet(
                dZlocal = cms.double(0.018),
                phiXlocal = cms.double(0.000412),
                dYlocal = cms.double(0.018),
                phiZlocal = cms.double(0.000412),
                dXlocal = cms.double(0.018),
                phiYlocal = cms.double(0.000412)
            )
        ),
        scaleError = cms.double(1.0),
        TIBLayer1_2 = cms.PSet(
            Dets = cms.PSet(
                dZlocal = cms.double(0.018),
                phiXlocal = cms.double(0.000412),
                dYlocal = cms.double(0.018),
                phiZlocal = cms.double(0.000412),
                dXlocal = cms.double(0.018),
                phiYlocal = cms.double(0.000412)
            )
        ),
        distribution = cms.string('gaussian'),
        TIBStrings = cms.PSet(
            dZlocal = cms.double(0.0275),
            phiXlocal = cms.double(0.000179),
            dYlocal = cms.double(0.0275),
            phiZlocal = cms.double(0.000179),
            dXlocal = cms.double(0.0275),
            phiYlocal = cms.double(0.000179)
        )
    ),
    TPBHalfBarrels = cms.PSet(
        scale = cms.double(1.0),
        DetUnits = cms.PSet(
            dZlocal = cms.double(0.006),
            phiXlocal = cms.double(0.00027),
            dYlocal = cms.double(0.006),
            phiZlocal = cms.double(0.00027),
            distribution = cms.string('gaussian'),
            dXlocal = cms.double(0.006),
            phiYlocal = cms.double(0.00027)
        ),
        TPBLayers = cms.PSet(
            phiXlocal = cms.double(9e-06),
            phiZlocal = cms.double(9e-06),
            dZ = cms.double(0.0174),
            dX = cms.double(0.0118),
            dY = cms.double(0.0118),
            distribution = cms.string('gaussian'),
            phiYlocal = cms.double(9e-06)
        ),
        scaleError = cms.double(1.0),
        TPBLadders = cms.PSet(
            dZlocal = cms.double(0.002),
            phiXlocal = cms.double(9e-06),
            dYlocal = cms.double(0.002),
            phiZlocal = cms.double(9e-06),
            distribution = cms.string('gaussian'),
            dXlocal = cms.double(0.002),
            phiYlocal = cms.double(9e-06)
        )
    ),
    TOBHalfBarrels = cms.PSet(
        scale = cms.double(1.0),
        TOBLayer3_4_5_6 = cms.PSet(
            DetUnits = cms.PSet(
                dZlocal = cms.double(0.0032),
                phiXlocal = cms.double(7.5e-05),
                dYlocal = cms.double(0.0032),
                phiZlocal = cms.double(7.5e-05),
                dXlocal = cms.double(0.0032),
                phiYlocal = cms.double(7.5e-05)
            )
        ),
        scaleError = cms.double(1.0),
        TOBLayers = cms.PSet(
            dXlocal = cms.double(0.0),
            dZlocal = cms.double(0.0),
            dYlocal = cms.double(0.0)
        ),
        TOBLayer1_2 = cms.PSet(
            Dets = cms.PSet(
                dZlocal = cms.double(0.0032),
                phiXlocal = cms.double(7.5e-05),
                dYlocal = cms.double(0.0032),
                phiZlocal = cms.double(7.5e-05),
                dXlocal = cms.double(0.0032),
                phiYlocal = cms.double(7.5e-05)
            )
        ),
        TOBRods = cms.PSet(
            dZlocal = cms.double(0.01),
            phiXlocal = cms.double(4e-05),
            dYlocal = cms.double(0.01),
            phiZlocal = cms.double(4e-05),
            dXlocal = cms.double(0.01),
            phiYlocal = cms.double(4e-05)
        ),
        TOBHalfBarrels = cms.PSet(
            dZlocal = cms.double(0.03),
            phiXlocal = cms.double(1e-05),
            dYlocal = cms.double(0.006),
            phiZlocal = cms.double(1e-05),
            dXlocal = cms.double(0.006),
            phiYlocal = cms.double(1e-05)
        ),
        distribution = cms.string('gaussian')
    ),
    TECEndcaps = cms.PSet(
        scale = cms.double(1.0),
        TECDisk9 = cms.PSet(
            TECRing2 = cms.PSet(
                TECRingsSurveyLASCosmics_2D
            ),
            TECRing1_3_4 = cms.PSet(
                TECRingsSurveyLASCosmics_1D
            )
        ),
        scaleError = cms.double(1.0),
        TECPetals = cms.PSet(
            dZlocal = cms.double(0.007),
            phiXlocal = cms.double(3e-05),
            dYlocal = cms.double(0.007),
            phiZlocal = cms.double(3e-05),
            dXlocal = cms.double(0.007),
            phiYlocal = cms.double(3e-05)
        ),
        TECDisks = cms.PSet(
            dZlocal = cms.double(0.015),
            phiXlocal = cms.double(1.5e-05),
            dYlocal = cms.double(0.006),
            phiZlocal = cms.double(1.5e-05),
            dXlocal = cms.double(0.006),
            phiYlocal = cms.double(1.5e-05)
        ),
        TECDisk4_5_6 = cms.PSet(
            TECRing2_3_5_6 = cms.PSet(
                TECRingsSurveyLASCosmics_1D
            ),
            TECRing1_4 = cms.PSet(
                TECRingsSurveyLASCosmics_2D
            )
        ),
        TECDisk7_8 = cms.PSet(
            TECRing3 = cms.PSet(
                TECRingsSurveyLASCosmics_2D
            ),
            TECRing1_2_4_5 = cms.PSet(
                TECRingsSurveyLASCosmics_1D
            )
        ),
        distribution = cms.string('gaussian'),
        TECDisk1_2_3 = cms.PSet(
            TECRing3_4_6_7 = cms.PSet(
                TECRingsSurveyLASCosmics_1D
            ),
            TECRing1_2_5 = cms.PSet(
                TECRingsSurveyLASCosmics_2D
            )
        )
    ),
    TPEEndcaps = cms.PSet(
        TPEPanels = cms.PSet(
            dZlocal = cms.double(0.001),
            phiXlocal = cms.double(0.0002),
            dYlocal = cms.double(0.001),
            phiZlocal = cms.double(0.0002),
            dXlocal = cms.double(0.001),
            phiYlocal = cms.double(0.0002)
        ),
        scale = cms.double(1.0),
        scaleError = cms.double(1.0),
        TPEHalfDisks = cms.PSet(
            dZlocal = cms.double(0.005),
            phiXlocal = cms.double(0.001),
            dYlocal = cms.double(0.005),
            phiZlocal = cms.double(0.001),
            dXlocal = cms.double(0.005),
            phiYlocal = cms.double(0.001)
        ),
        TPEHalfCylinders = cms.PSet(
            dZlocal = cms.double(0.005),
            phiXlocal = cms.double(0.001),
            dYlocal = cms.double(0.005),
            phiZlocal = cms.double(0.001),
            dXlocal = cms.double(0.005),
            phiYlocal = cms.double(0.001)
        ),
        DetUnits = cms.PSet(
            dZlocal = cms.double(0.0005),
            phiXlocal = cms.double(0.0001),
            dYlocal = cms.double(0.0005),
            phiZlocal = cms.double(0.0001),
            dXlocal = cms.double(0.0005),
            phiYlocal = cms.double(0.0001)
        ),
        TPEBlades = cms.PSet(
            dZlocal = cms.double(0.001),
            phiXlocal = cms.double(0.0002),
            dYlocal = cms.double(0.001),
            phiZlocal = cms.double(0.0002),
            dXlocal = cms.double(0.001),
            phiYlocal = cms.double(0.0002)
        ),
        distribution = cms.string('gaussian')
    ),
    TIDEndcaps = cms.PSet(
        scale = cms.double(1.0),
        scaleError = cms.double(1.0),
        TIDEndcaps = cms.PSet(
            dZlocal = cms.double(0.0225),
            phiXlocal = cms.double(0.000325),
            dYlocal = cms.double(0.0225),
            phiZlocal = cms.double(0.000325),
            dXlocal = cms.double(0.0225),
            phiYlocal = cms.double(0.000325)
        ),
        TIDRing3 = cms.PSet(
            DetUnits = cms.PSet(
                dZlocal = cms.double(0.0054),
                phiXlocal = cms.double(0.00025),
                dYlocal = cms.double(0.0054),
                phiZlocal = cms.double(0.00025),
                dXlocal = cms.double(0.0054),
                phiYlocal = cms.double(0.00025)
            )
        ),
        TIDRing1_2 = cms.PSet(
            Dets = cms.PSet(
                dZlocal = cms.double(0.0054),
                phiXlocal = cms.double(0.00025),
                dYlocal = cms.double(0.0054),
                phiZlocal = cms.double(0.00025),
                dXlocal = cms.double(0.0054),
                phiYlocal = cms.double(0.00025)
            )
        ),
        TIDDisks = cms.PSet(
            dZlocal = cms.double(0.03),
            phiXlocal = cms.double(0.000456),
            dYlocal = cms.double(0.03),
            phiZlocal = cms.double(0.000456),
            dXlocal = cms.double(0.03),
            phiYlocal = cms.double(0.000456)
        ),
        distribution = cms.string('gaussian'),
        TIDRings = cms.PSet(
            dZlocal = cms.double(0.0185),
            phiXlocal = cms.double(0.00085),
            dYlocal = cms.double(0.0185),
            phiZlocal = cms.double(0.00085),
            dXlocal = cms.double(0.0185),
            phiYlocal = cms.double(0.00085)
        )
    )
)
# -----------------------------------------------------------------------
# TrackerNoKnowledge scenario
# ----------------------------------------
# ----------------- NOTE -----------------
# This scenario is not supposed to be used
# to make public(?) estimates of the 
# performance of the CMS.  
#
# This scenario contains lots of guesses,
# and is intended to be used as a stress 
# test for aligning with cosmics as well
# as a initial misalignment for LAS studies
# ------------- NOTE ends ----------------
# ----------------------------------------
# first helper blocks for TEC:
TECRingsTrackerNoKnowledge_1D = cms.PSet(
    DetUnits = cms.PSet(
        dZlocal = cms.double(0.0054),
        phiXlocal = cms.double(0.0005),
        dYlocal = cms.double(0.0028),
        phiZlocal = cms.double(0.0003),
        dXlocal = cms.double(0.0028),
        phiYlocal = cms.double(0.0005)
    )
)
TECRingsTrackerNoKnowledge_2D = cms.PSet(
    Dets = cms.PSet(
        dZlocal = cms.double(0.0054),
        phiXlocal = cms.double(0.0005),
        dYlocal = cms.double(0.0028),
        phiZlocal = cms.double(0.0003),
        dXlocal = cms.double(0.0028),
        phiYlocal = cms.double(0.0005)
    )
)
# now actual scenario
TrackerNoKnowledgeScenario = cms.PSet(
    MisalignmentScenarioSettings,
    TIBHalfBarrels = cms.PSet(
        TIBLayers = cms.PSet(
            dZlocal = cms.double(0.02),
            phiXlocal = cms.double(0.0006),
            dYlocal = cms.double(0.04),
            phiZlocal = cms.double(0.0006),
            dXlocal = cms.double(0.04),
            phiYlocal = cms.double(0.0006)
        ),
        scale = cms.double(1.0),
        TIBLayer3_4 = cms.PSet(
            DetUnits = cms.PSet(
                dZlocal = cms.double(0.005),
                phiXlocal = cms.double(0.001),
                dYlocal = cms.double(0.005),
                phiZlocal = cms.double(0.0005),
                dXlocal = cms.double(0.005),
                phiYlocal = cms.double(0.0005)
            )
        ),
        scaleError = cms.double(1.0),
        TIBHalfShells = cms.PSet(
            dZlocal = cms.double(0.015),
            phiXlocal = cms.double(0.0004),
            dYlocal = cms.double(0.03),
            phiZlocal = cms.double(0.0004),
            dXlocal = cms.double(0.03),
            phiYlocal = cms.double(0.0004)
        ),
        TIBBarrels = cms.PSet( # FIXME??
            dZlocal = cms.double(0.2),
            phiXlocal = cms.double(0.0017),
            dYlocal = cms.double(0.2),
            phiZlocal = cms.double(0.0017),
            dXlocal = cms.double(0.2),
            phiYlocal = cms.double(0.0017)
        ),
        TIBHalfBarrels = cms.PSet( # FIXME??
            dZlocal = cms.double(0.1),
            phiXlocal = cms.double(0.0008),
            dYlocal = cms.double(0.1),
            phiZlocal = cms.double(0.0008),
            dXlocal = cms.double(0.1),
            phiYlocal = cms.double(0.0008)
        ),
        TIBLayer1_2 = cms.PSet(
            Dets = cms.PSet(
                dZlocal = cms.double(0.005),
                phiXlocal = cms.double(0.001),
                dYlocal = cms.double(0.005),
                phiZlocal = cms.double(0.0005),
                dXlocal = cms.double(0.005),
                phiYlocal = cms.double(0.0005)
            )
        ),
        distribution = cms.string('gaussian'),
        TIBStrings = cms.PSet(
            dZlocal = cms.double(0.01),
            phiXlocal = cms.double(0.0004),
            dYlocal = cms.double(0.01),
            phiZlocal = cms.double(0.0002),
            dXlocal = cms.double(0.01),
            phiYlocal = cms.double(0.0002)
        ),
        TIBSurfaces = cms.PSet(
            dZlocal = cms.double(0.01),
            phiXlocal = cms.double(0.0004),
            dYlocal = cms.double(0.01),
            phiZlocal = cms.double(0.0002),
            dXlocal = cms.double(0.01),
            phiYlocal = cms.double(0.0002)
        )
    ),
    TPBHalfBarrels = cms.PSet(
        TPBHalfBarrels = cms.PSet( # FIXME??
            phiXlocal = cms.double(0.0008),
            phiZlocal = cms.double(0.0008),
            dZ = cms.double(0.05),
            dX = cms.double(0.05),
            dY = cms.double(0.05),
            distribution = cms.string('gaussian'),
            phiYlocal = cms.double(0.0008)
        ),
        scale = cms.double(1.0),
        scaleError = cms.double(1.0),
        TPBLadders = cms.PSet(
            dZlocal = cms.double(0.005),
            phiXlocal = cms.double(0.0002),
            dYlocal = cms.double(0.01),
            phiZlocal = cms.double(0.0002),
            distribution = cms.string('gaussian'),
            dXlocal = cms.double(0.01),
            phiYlocal = cms.double(0.0002)
        ),
        DetUnits = cms.PSet(
            dZlocal = cms.double(0.003),
            phiXlocal = cms.double(0.001),
            dYlocal = cms.double(0.005),
            phiZlocal = cms.double(0.001),
            distribution = cms.string('gaussian'),
            dXlocal = cms.double(0.006),
            phiYlocal = cms.double(0.001)
        ),
        TPBLayers = cms.PSet(
            phiXlocal = cms.double(0.0004),
            phiZlocal = cms.double(0.0004),
            dZ = cms.double(0.02),
            dX = cms.double(0.02),
            dY = cms.double(0.02),
            distribution = cms.string('gaussian'),
            phiYlocal = cms.double(0.0004)
        ),
        TPBBarrels = cms.PSet( # FIXME??
            phiXlocal = cms.double(0.003),
            phiZlocal = cms.double(0.003),
            dZ = cms.double(0.1),
            dX = cms.double(0.1),
            dY = cms.double(0.1),
            distribution = cms.string('gaussian'),
            phiYlocal = cms.double(0.003)
        )
    ),
    TOBHalfBarrels = cms.PSet(
        scale = cms.double(1.0),
        TOBBarrels = cms.PSet( # FIXME??
            dZlocal = cms.double(0.1),
            phiXlocal = cms.double(0.0005),
            dYlocal = cms.double(0.1),
            phiZlocal = cms.double(0.0005),
            dXlocal = cms.double(0.1),
            phiYlocal = cms.double(0.0005)
        ),
        scaleError = cms.double(1.0),
        TOBLayers = cms.PSet(
            dZlocal = cms.double(0.02),
            phiXlocal = cms.double(0.0001),
            dYlocal = cms.double(0.01),
            phiZlocal = cms.double(0.0001),
            dXlocal = cms.double(0.01),
            phiYlocal = cms.double(0.0001)
        ),
        TOBLayer1_2 = cms.PSet(
            Dets = cms.PSet(
                dZlocal = cms.double(0.01),
                phiXlocal = cms.double(0.0002),
                dYlocal = cms.double(0.003),
                phiZlocal = cms.double(0.0002),
                dXlocal = cms.double(0.003),
                phiYlocal = cms.double(0.0002)
            )
        ),
        TOBRods = cms.PSet(
            dZlocal = cms.double(0.01),
            phiXlocal = cms.double(0.0001),
            dYlocal = cms.double(0.02),
            phiZlocal = cms.double(0.0001),
            dXlocal = cms.double(0.01),
            phiYlocal = cms.double(0.0002)
        ),
        TOBHalfBarrels = cms.PSet( # FIXME??
            dZlocal = cms.double(0.02),
            phiXlocal = cms.double(0.0001),
            dYlocal = cms.double(0.01),
            phiZlocal = cms.double(0.0001),
            dXlocal = cms.double(0.01),
            phiYlocal = cms.double(0.0001)
        ),
        distribution = cms.string('gaussian'),
        TOBLayer3_4_5_6 = cms.PSet(
            DetUnits = cms.PSet(
                dZlocal = cms.double(0.01),
                phiXlocal = cms.double(0.0002),
                dYlocal = cms.double(0.003),
                phiZlocal = cms.double(0.0002),
                dXlocal = cms.double(0.003),
                phiYlocal = cms.double(0.0002)
            )
        )
    ),
    TECEndcaps = cms.PSet(
        scale = cms.double(1.0),
        TECDisk9 = cms.PSet(
            TECRing2 = cms.PSet(
                TECRingsTrackerNoKnowledge_2D
            ),
            TECRing1_3_4 = cms.PSet(
                TECRingsTrackerNoKnowledge_1D
            )
        ),
        TECDisk1_2_3 = cms.PSet(
            TECRing3_4_6_7 = cms.PSet(
                TECRingsTrackerNoKnowledge_1D
            ),
            TECRing1_2_5 = cms.PSet(
                TECRingsTrackerNoKnowledge_2D
            )
        ),
        scaleError = cms.double(1.0),
        TECEndcaps = cms.PSet( # FIXME??
            dZlocal = cms.double(0.0316),
            phiXlocal = cms.double(0.001),
            dYlocal = cms.double(0.05),
            phiZlocal = cms.double(0.001),
            dXlocal = cms.double(0.0316),
            phiYlocal = cms.double(0.001)
        ),
        TECDisk4_5_6 = cms.PSet(
            TECRing2_3_5_6 = cms.PSet(
                TECRingsTrackerNoKnowledge_1D
            ),
            TECRing1_4 = cms.PSet(
                TECRingsTrackerNoKnowledge_2D
            )
        ),
        TECPetals = cms.PSet(
            dZlocal = cms.double(0.0158),
            phiXlocal = cms.double(0.0005),
            dYlocal = cms.double(0.0086),
            phiZlocal = cms.double(0.0003),
            dXlocal = cms.double(0.0086),
            phiYlocal = cms.double(0.0005)
        ),
        TECDisks = cms.PSet(
            dZlocal = cms.double(0.0112),
            phiXlocal = cms.double(0.0002),
            dYlocal = cms.double(0.0206),
            phiZlocal = cms.double(0.0001),
            dXlocal = cms.double(0.0112),
            phiYlocal = cms.double(0.0001)
        ),
        TECDisk7_8 = cms.PSet(
            TECRing3 = cms.PSet(
                TECRingsTrackerNoKnowledge_2D
            ),
            TECRing1_2_4_5 = cms.PSet(
                TECRingsTrackerNoKnowledge_1D
            )
        ),
        distribution = cms.string('gaussian')
    ),
    TPEEndcaps = cms.PSet(
        TPEPanels = cms.PSet(
            dZlocal = cms.double(0.005),
            phiXlocal = cms.double(0.002),
            dYlocal = cms.double(0.01),
            phiZlocal = cms.double(0.002),
            dXlocal = cms.double(0.01),
            phiYlocal = cms.double(0.002)
        ),
        scale = cms.double(1.0),
        scaleError = cms.double(1.0),
        TPEHalfDisks = cms.PSet(
            dZlocal = cms.double(0.02),
            phiXlocal = cms.double(0.001),
            dYlocal = cms.double(0.02),
            phiZlocal = cms.double(0.001),
            dXlocal = cms.double(0.02),
            phiYlocal = cms.double(0.001)
        ),
        TPEEndcaps = cms.PSet( # FIXME??
            dZlocal = cms.double(0.1),
            phiXlocal = cms.double(0.0017),
            dYlocal = cms.double(0.1),
            phiZlocal = cms.double(0.0017),
            dXlocal = cms.double(0.1),
            phiYlocal = cms.double(0.0017)
        ),
        TPEHalfCylinders = cms.PSet(
            dZlocal = cms.double(0.05),
            phiXlocal = cms.double(0.0004),
            dYlocal = cms.double(0.05),
            phiZlocal = cms.double(0.0004),
            dXlocal = cms.double(0.05),
            phiYlocal = cms.double(0.0004)
        ),
        DetUnits = cms.PSet(
            dZlocal = cms.double(0.003),
            phiXlocal = cms.double(0.001),
            dYlocal = cms.double(0.005),
            phiZlocal = cms.double(0.001),
            dXlocal = cms.double(0.005),
            phiYlocal = cms.double(0.001)
        ),
        TPEBlades = cms.PSet(
            dZlocal = cms.double(0.01),
            phiXlocal = cms.double(0.001),
            dYlocal = cms.double(0.01),
            phiZlocal = cms.double(0.001),
            dXlocal = cms.double(0.01),
            phiYlocal = cms.double(0.001)
        ),
        distribution = cms.string('gaussian')
    ),
    TIDEndcaps = cms.PSet(
        TIDSides = cms.PSet(
            dZlocal = cms.double(0.005),
            phiXlocal = cms.double(0.0001),
            dYlocal = cms.double(0.005),
            phiZlocal = cms.double(0.0001),
            dXlocal = cms.double(0.005),
            phiYlocal = cms.double(0.0001)
        ),
        scale = cms.double(1.0),
        scaleError = cms.double(1.0),
        TIDEndcaps = cms.PSet( # FIXME??
            dZlocal = cms.double(0.1),
            phiXlocal = cms.double(0.0013),
            dYlocal = cms.double(0.1),
            phiZlocal = cms.double(0.0013),
            dXlocal = cms.double(0.1),
            phiYlocal = cms.double(0.0013)
        ),
        TIDRings = cms.PSet(
            dZlocal = cms.double(0.01),
            phiXlocal = cms.double(0.0001),
            dYlocal = cms.double(0.01),
            phiZlocal = cms.double(0.0001),
            dXlocal = cms.double(0.01),
            phiYlocal = cms.double(0.0001)
        ),
        TIDRing1_2 = cms.PSet(
            Dets = cms.PSet(
                dZlocal = cms.double(0.005),
                phiXlocal = cms.double(0.0005),
                dYlocal = cms.double(0.005),
                phiZlocal = cms.double(0.0005),
                dXlocal = cms.double(0.005),
                phiYlocal = cms.double(0.0005)
            )
        ),
        TIDDisks = cms.PSet(
            dZlocal = cms.double(0.05),
            phiXlocal = cms.double(0.0004),
            dYlocal = cms.double(0.05),
            phiZlocal = cms.double(0.0004),
            dXlocal = cms.double(0.05),
            phiYlocal = cms.double(0.0004)
        ),
        distribution = cms.string('gaussian'),
        TIDRing3 = cms.PSet(
            DetUnits = cms.PSet(
                dZlocal = cms.double(0.005),
                phiXlocal = cms.double(0.0005),
                dYlocal = cms.double(0.005),
                phiZlocal = cms.double(0.0005),
                dXlocal = cms.double(0.005),
                phiYlocal = cms.double(0.0005)
            )
        )
    )
)



# -----------------------------------------------------------------------
# TrackerAfterCRAFTScenario
# ----------------------------------------
# ----------------- NOTE -----------------
# This scenario is a simple merge of scenarios above:
# It seems that after CRAFT alignment we have aligned TIB and TOB
# as good as foreseen after 100 pb^-1 (at least what concerns residuals [DMR]),
# while the rest of the tracker is at the level of the 10 pb^-1 scenario
# or even still at that of SurveyLASOnlyScenario.
# ------------- NOTE ends ----------------
# ----------------------------------------
TrackerCRAFTScenario = copy.deepcopy(Tracker10pbScenario)
TrackerCRAFTScenario.TIBHalfBarrels = copy.deepcopy(Tracker100pbScenario.TIBHalfBarrels)
TrackerCRAFTScenario.TOBHalfBarrels = copy.deepcopy(Tracker100pbScenario.TOBHalfBarrels)
TrackerCRAFTScenario.TPEEndcaps = copy.deepcopy(TrackerSurveyLASOnlyScenario.TPEEndcaps)
