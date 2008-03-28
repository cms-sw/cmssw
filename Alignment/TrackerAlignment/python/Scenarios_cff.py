# The following comments couldn't be translated into the new config version:

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

import FWCore.ParameterSet.Config as cms

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
    saveToDbase = cms.untracked.bool(False),
    setRotations = cms.bool(True),
    setTranslations = cms.bool(True),
    seed = cms.int32(1234567),
    distribution = cms.string('gaussian'),
    setError = cms.bool(True)
)
# -----------------------------------------------------------------------
# Example scenario (dummy movements)
TrackerExampleScenario = cms.PSet(
    MisalignmentScenarioSettings,
    TOB1 = cms.PSet(
        TOBLayers = cms.PSet(
            dX = cms.double(0.1)
        ),
        TOBLayer1 = cms.PSet(
            dX = cms.double(0.2)
        )
    ),
    TOBs = cms.PSet(
        TOBLayer1 = cms.PSet(
            phiX = cms.double(0.03)
        ),
        dX = cms.double(0.2)
    )
)
# -----------------------------------------------------------------------
#  "Misalignment" scenario without misalignment...
NoMovementsScenario = cms.PSet(
    MisalignmentScenarioSettings
)
# -----------------------------------------------------------------------
# Short term misalignment scenario as it is (wrongly) implemented in ORCA
# Layer movements applied at structure level instead of layer level
obsoleteTrackerORCAShortTermScenario = cms.PSet(
    MisalignmentScenarioSettings,
    TIBs = cms.PSet(
        scale = cms.double(1.0),
        distribution = cms.string('gaussian'),
        dZ = cms.double(0.05),
        dX = cms.double(0.0105),
        dY = cms.double(0.0105),
        Dets = cms.PSet(
            distribution = cms.string('flat'),
            dZ = cms.double(0.02),
            dX = cms.double(0.02),
            dY = cms.double(0.02)
        ),
        TIBStrings = cms.PSet(
            distribution = cms.string('flat'),
            dZ = cms.double(0.02),
            dX = cms.double(0.02),
            dY = cms.double(0.02)
        ),
        phiZ = cms.double(9e-05)
    ),
    TPBs = cms.PSet(
        scale = cms.double(1.0),
        distribution = cms.string('gaussian'),
        TPBLadders = cms.PSet(
            dZ = cms.double(0.0005),
            dX = cms.double(0.0005),
            dY = cms.double(0.0005)
        ),
        dZ = cms.double(0.001),
        dX = cms.double(0.001),
        dY = cms.double(0.001),
        Dets = cms.PSet(
            dZ = cms.double(0.0013),
            dX = cms.double(0.0013),
            dY = cms.double(0.0013)
        ),
        phiZ = cms.double(1e-05)
    ),
    TOBs = cms.PSet(
        scale = cms.double(1.0),
        TOBRods = cms.PSet(
            distribution = cms.string('flat'),
            dZ = cms.double(0.01),
            dX = cms.double(0.01),
            dY = cms.double(0.01)
        ),
        distribution = cms.string('gaussian'),
        dZ = cms.double(0.05),
        dX = cms.double(0.0067),
        dY = cms.double(0.0067),
        Dets = cms.PSet(
            distribution = cms.string('flat'),
            dZ = cms.double(0.01),
            dX = cms.double(0.01),
            dY = cms.double(0.01)
        ),
        phiZ = cms.double(5.9e-05)
    ),
    TECs = cms.PSet(
        scale = cms.double(1.0),
        TECPetals = cms.PSet(
            distribution = cms.string('flat'),
            dZ = cms.double(0.01),
            dX = cms.double(0.01),
            dY = cms.double(0.01)
        ),
        distribution = cms.string('gaussian'),
        dZ = cms.double(0.05),
        dX = cms.double(0.0057),
        dY = cms.double(0.0057),
        Dets = cms.PSet(
            distribution = cms.string('flat'),
            dZ = cms.double(0.005),
            dX = cms.double(0.005),
            dY = cms.double(0.005)
        ),
        phiZ = cms.double(4.6e-05)
    ),
    TPEs = cms.PSet(
        TPEPanels = cms.PSet(
            dZ = cms.double(0.00025),
            dX = cms.double(0.00025),
            dY = cms.double(0.00025)
        ),
        scale = cms.double(1.0),
        distribution = cms.string('gaussian'),
        dZ = cms.double(0.0005),
        TPEBlades = cms.PSet(
            dZ = cms.double(0.0),
            dX = cms.double(0.0),
            dY = cms.double(0.0)
        ),
        dX = cms.double(0.0005),
        dY = cms.double(0.0005),
        Dets = cms.PSet(
            dZ = cms.double(0.0005),
            dX = cms.double(0.0005),
            dY = cms.double(0.0005)
        ),
        phiZ = cms.double(5e-06)
    ),
    TIDs = cms.PSet(
        scale = cms.double(1.0),
        distribution = cms.string('flat'),
        dZ = cms.double(0.04),
        dX = cms.double(0.04),
        dY = cms.double(0.04),
        Dets = cms.PSet(
            dZ = cms.double(0.0105),
            dX = cms.double(0.0105),
            dY = cms.double(0.0105)
        ),
        TIDRings = cms.PSet(
            dZ = cms.double(0.03),
            dX = cms.double(0.03),
            dY = cms.double(0.03)
        ),
        phiZ = cms.double(0.0001)
    )
)
# -----------------------------------------------------------------------
# Short term misalignment scenario (as interpreted from AN-2005-036)
# Layer movements applied at layer level
obsoleteTrackerShortTermScenario = cms.PSet(
    MisalignmentScenarioSettings,
    TIBs = cms.PSet(
        Dets = cms.PSet(
            distribution = cms.string('flat'),
            dZ = cms.double(0.02),
            dX = cms.double(0.02),
            dY = cms.double(0.02)
        ),
        TIBStrings = cms.PSet(
            distribution = cms.string('flat'),
            dZ = cms.double(0.02),
            dX = cms.double(0.02),
            dY = cms.double(0.02)
        ),
        scale = cms.double(1.0),
        TIBLayers = cms.PSet(
            distribution = cms.string('gaussian'),
            dZ = cms.double(0.05),
            phiZ = cms.double(9e-05),
            dX = cms.double(0.0105),
            dY = cms.double(0.0105)
        )
    ),
    TPBs = cms.PSet(
        TPBLadders = cms.PSet(
            dZ = cms.double(0.0005),
            dX = cms.double(0.0005),
            dY = cms.double(0.0005)
        ),
        Dets = cms.PSet(
            dZ = cms.double(0.0013),
            dX = cms.double(0.0013),
            dY = cms.double(0.0013)
        ),
        scale = cms.double(1.0),
        distribution = cms.string('gaussian'),
        TPBLayers = cms.PSet(
            dZ = cms.double(0.001),
            phiZ = cms.double(1e-05),
            dX = cms.double(0.001),
            dY = cms.double(0.001)
        )
    ),
    TOBs = cms.PSet(
        TOBLayers = cms.PSet(
            distribution = cms.string('gaussian'),
            dZ = cms.double(0.05),
            phiZ = cms.double(5.9e-05),
            dX = cms.double(0.0067),
            dY = cms.double(0.0067)
        ),
        Dets = cms.PSet(
            distribution = cms.string('flat'),
            dZ = cms.double(0.01),
            dX = cms.double(0.01),
            dY = cms.double(0.01)
        ),
        scale = cms.double(1.0),
        TOBRods = cms.PSet(
            distribution = cms.string('flat'),
            dZ = cms.double(0.01),
            dX = cms.double(0.01),
            dY = cms.double(0.01)
        )
    ),
    TECs = cms.PSet(
        Dets = cms.PSet(
            distribution = cms.string('flat'),
            dZ = cms.double(0.005),
            dX = cms.double(0.005),
            dY = cms.double(0.005)
        ),
        scale = cms.double(1.0),
        TECDisks = cms.PSet(
            distribution = cms.string('gaussian'),
            dZ = cms.double(0.05),
            phiZ = cms.double(4.6e-05),
            dX = cms.double(0.0057),
            dY = cms.double(0.0057)
        ),
        TECPetals = cms.PSet(
            distribution = cms.string('flat'),
            dZ = cms.double(0.01),
            dX = cms.double(0.01),
            dY = cms.double(0.01)
        )
    ),
    TPEs = cms.PSet(
        TPEPanels = cms.PSet(
            dZ = cms.double(0.00018),
            dX = cms.double(0.00018),
            dY = cms.double(0.00018)
        ),
        scale = cms.double(1.0),
        distribution = cms.string('gaussian'),
        TPEBlades = cms.PSet(
            dZ = cms.double(0.00018),
            dX = cms.double(0.00018),
            dY = cms.double(0.00018)
        ),
        Dets = cms.PSet(
            dZ = cms.double(0.0005),
            dX = cms.double(0.0005),
            dY = cms.double(0.0005)
        ),
        TPEHalfDisk = cms.PSet(
            dZ = cms.double(0.0005),
            phiZ = cms.double(5e-06),
            dX = cms.double(0.0005),
            dY = cms.double(0.0005)
        )
    ),
    TIDs = cms.PSet(
        Dets = cms.PSet(
            dZ = cms.double(0.0105),
            dX = cms.double(0.0105),
            dY = cms.double(0.0105)
        ),
        TIDDisks = cms.PSet(
            dZ = cms.double(0.04),
            phiZ = cms.double(0.0001),
            dX = cms.double(0.04),
            dY = cms.double(0.04)
        ),
        TIDRings = cms.PSet(
            dZ = cms.double(0.03),
            dX = cms.double(0.03),
            dY = cms.double(0.03)
        ),
        distribution = cms.string('flat'),
        scale = cms.double(1.0)
    )
)
# -----------------------------------------------------------------------
# Long term misalignment scenario (as interpreted from AN-2005-036)
# Layer movements applied at layer level
obsoleteTrackerLongTermScenario = cms.PSet(
    MisalignmentScenarioSettings,
    TIBs = cms.PSet(
        Dets = cms.PSet(
            dZ = cms.double(0.002),
            dX = cms.double(0.002),
            dY = cms.double(0.002)
        ),
        TIBStrings = cms.PSet(
            dZ = cms.double(0.002),
            dX = cms.double(0.002),
            dY = cms.double(0.002)
        ),
        scale = cms.double(1.0),
        TIBLayers = cms.PSet(
            dZ = cms.double(0.005),
            phiZ = cms.double(9e-06),
            dX = cms.double(0.00105),
            dY = cms.double(0.00105)
        )
    ),
    TPBs = cms.PSet(
        Dets = cms.PSet(
            dZ = cms.double(0.0013),
            dX = cms.double(0.0013),
            dY = cms.double(0.0013)
        ),
        scale = cms.double(1.0),
        TPBLayers = cms.PSet(
            dZ = cms.double(0.001),
            phiZ = cms.double(1e-05),
            dX = cms.double(0.001),
            dY = cms.double(0.001)
        ),
        TPBLadders = cms.PSet(
            dZ = cms.double(0.0005),
            dX = cms.double(0.0005),
            dY = cms.double(0.0005)
        )
    ),
    TOBs = cms.PSet(
        TOBLayers = cms.PSet(
            dZ = cms.double(0.005),
            phiZ = cms.double(5.9e-06),
            dX = cms.double(0.00067),
            dY = cms.double(0.00067)
        ),
        Dets = cms.PSet(
            dZ = cms.double(0.001),
            dX = cms.double(0.001),
            dY = cms.double(0.001)
        ),
        scale = cms.double(1.0),
        TOBRods = cms.PSet(
            dZ = cms.double(0.001),
            dX = cms.double(0.001),
            dY = cms.double(0.001)
        )
    ),
    TECs = cms.PSet(
        Dets = cms.PSet(
            dZ = cms.double(0.0005),
            dX = cms.double(0.0005),
            dY = cms.double(0.0005)
        ),
        scale = cms.double(1.0),
        TECDisks = cms.PSet(
            dZ = cms.double(0.005),
            phiZ = cms.double(4.6e-06),
            dX = cms.double(0.00057),
            dY = cms.double(0.00057)
        ),
        TECPetals = cms.PSet(
            dZ = cms.double(0.001),
            dX = cms.double(0.001),
            dY = cms.double(0.001)
        )
    ),
    TPEs = cms.PSet(
        TPEPanels = cms.PSet(
            dZ = cms.double(0.00018),
            dX = cms.double(0.00018),
            dY = cms.double(0.00018)
        ),
        Dets = cms.PSet(
            dZ = cms.double(0.0005),
            dX = cms.double(0.0005),
            dY = cms.double(0.0005)
        ),
        scale = cms.double(1.0),
        TPEBlades = cms.PSet(
            dZ = cms.double(0.00018),
            dX = cms.double(0.00018),
            dY = cms.double(0.00018)
        ),
        TPEHalfDisk = cms.PSet(
            dZ = cms.double(0.0005),
            phiZ = cms.double(5e-06),
            dX = cms.double(0.0005),
            dY = cms.double(0.0005)
        )
    ),
    TIDs = cms.PSet(
        Dets = cms.PSet(
            dZ = cms.double(0.00105),
            dX = cms.double(0.00105),
            dY = cms.double(0.00105)
        ),
        TIDDisks = cms.PSet(
            dZ = cms.double(0.004),
            phiZ = cms.double(1e-05),
            dX = cms.double(0.004),
            dY = cms.double(0.004)
        ),
        TIDRings = cms.PSet(
            dZ = cms.double(0.003),
            dX = cms.double(0.003),
            dY = cms.double(0.003)
        ),
        scale = cms.double(1.0)
    )
)
# -----------------------------------------------------------------------
# 10 pb-1 misalignment scenario
# See CMS IN 2007/036
Tracker10pbScenario = cms.PSet(
    MisalignmentScenarioSettings,
    TIBs = cms.PSet(
        Dets = cms.PSet(
            dZlocal = cms.double(0.018),
            phiXlocal = cms.double(0.000412),
            dYlocal = cms.double(0.018),
            phiZlocal = cms.double(0.000412),
            dXlocal = cms.double(0.018),
            phiYlocal = cms.double(0.000412)
        ),
        TIBStrings = cms.PSet(
            dZlocal = cms.double(0.01),
            phiXlocal = cms.double(6.5e-05),
            dYlocal = cms.double(0.01),
            phiZlocal = cms.double(6.5e-05),
            dXlocal = cms.double(0.01),
            phiYlocal = cms.double(6.5e-05)
        ),
        distribution = cms.string('gaussian'),
        TIBLayers = cms.PSet(
            dZlocal = cms.double(0.01),
            phiXlocal = cms.double(6.5e-05),
            dYlocal = cms.double(0.01),
            phiZlocal = cms.double(6.5e-05),
            dXlocal = cms.double(0.01),
            phiYlocal = cms.double(6.5e-05)
        ),
        scale = cms.double(1.0)
    ),
    TPBs = cms.PSet(
        Dets = cms.PSet(
            dZlocal = cms.double(0.006),
            phiXlocal = cms.double(0.00027),
            dYlocal = cms.double(0.006),
            phiZlocal = cms.double(0.00027),
            distribution = cms.string('gaussian'),
            dXlocal = cms.double(0.006),
            phiYlocal = cms.double(0.00027)
        ),
        scale = cms.double(1.0),
        TPBLayers = cms.PSet(
            phiXlocal = cms.double(7e-06),
            phiZlocal = cms.double(7e-06),
            dZ = cms.double(0.001),
            dX = cms.double(0.001),
            dY = cms.double(0.001),
            distribution = cms.string('gaussian'),
            phiYlocal = cms.double(7e-06)
        ),
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
    TOBs = cms.PSet(
        scale = cms.double(1.0),
        TOBLayers = cms.PSet(
            dXlocal = cms.double(0.0),
            dZlocal = cms.double(0.0),
            dYlocal = cms.double(0.0)
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
        distribution = cms.string('gaussian'),
        Dets = cms.PSet(
            dZlocal = cms.double(0.0032),
            phiXlocal = cms.double(7.5e-05),
            dYlocal = cms.double(0.0032),
            phiZlocal = cms.double(7.5e-05),
            dXlocal = cms.double(0.0032),
            phiYlocal = cms.double(7.5e-05)
        )
    ),
    TECs = cms.PSet(
        TECDisks = cms.PSet(
            dZlocal = cms.double(0.015),
            phiXlocal = cms.double(1.5e-05),
            dYlocal = cms.double(0.006),
            phiZlocal = cms.double(1.5e-05),
            dXlocal = cms.double(0.006),
            phiYlocal = cms.double(1.5e-05)
        ),
        distribution = cms.string('gaussian'),
        scale = cms.double(1.0),
        Dets = cms.PSet(
            dZlocal = cms.double(0.0022),
            phiXlocal = cms.double(5e-05),
            dYlocal = cms.double(0.0022),
            phiZlocal = cms.double(5e-05),
            dXlocal = cms.double(0.0022),
            phiYlocal = cms.double(5e-05)
        ),
        TECPetals = cms.PSet(
            dZlocal = cms.double(0.007),
            phiXlocal = cms.double(3e-05),
            dYlocal = cms.double(0.007),
            phiZlocal = cms.double(3e-05),
            dXlocal = cms.double(0.007),
            phiYlocal = cms.double(3e-05)
        )
    ),
    TPEs = cms.PSet(
        TPEPanels = cms.PSet(
            dZlocal = cms.double(0.001),
            phiXlocal = cms.double(0.0002),
            dYlocal = cms.double(0.001),
            phiZlocal = cms.double(0.0002),
            dXlocal = cms.double(0.001),
            phiYlocal = cms.double(0.0002)
        ),
        scale = cms.double(1.0),
        TPEHalfDisks = cms.PSet(
            dZlocal = cms.double(0.001),
            phiXlocal = cms.double(1.5e-05),
            dYlocal = cms.double(0.001),
            phiZlocal = cms.double(1.5e-05),
            dXlocal = cms.double(0.001),
            phiYlocal = cms.double(1.5e-05)
        ),
        TPEBlades = cms.PSet(
            dZlocal = cms.double(0.001),
            phiXlocal = cms.double(1.5e-05),
            dYlocal = cms.double(0.001),
            phiZlocal = cms.double(1.5e-05),
            dXlocal = cms.double(0.001),
            phiYlocal = cms.double(1.5e-05)
        ),
        distribution = cms.string('gaussian'),
        Dets = cms.PSet(
            dZlocal = cms.double(0.0005),
            phiXlocal = cms.double(0.0001),
            dYlocal = cms.double(0.0005),
            phiZlocal = cms.double(0.0001),
            dXlocal = cms.double(0.0005),
            phiYlocal = cms.double(0.0001)
        )
    ),
    TIDs = cms.PSet(
        Dets = cms.PSet(
            dZlocal = cms.double(0.0054),
            phiXlocal = cms.double(0.00025),
            dYlocal = cms.double(0.0054),
            phiZlocal = cms.double(0.00025),
            dXlocal = cms.double(0.0054),
            phiYlocal = cms.double(0.00025)
        ),
        TIDDisks = cms.PSet(
            dZlocal = cms.double(0.025),
            phiXlocal = cms.double(0.00038),
            dYlocal = cms.double(0.025),
            phiZlocal = cms.double(0.00038),
            dXlocal = cms.double(0.025),
            phiYlocal = cms.double(0.00038)
        ),
        TIDRings = cms.PSet(
            dZlocal = cms.double(0.0185),
            phiXlocal = cms.double(0.00085),
            dYlocal = cms.double(0.0185),
            phiZlocal = cms.double(0.00085),
            dXlocal = cms.double(0.0185),
            phiYlocal = cms.double(0.00085)
        ),
        distribution = cms.string('gaussian'),
        scale = cms.double(1.0)
    )
)
# 100 pb-1 misalignment scenario
# See CMS IN 2007/036
Tracker100pbScenario = cms.PSet(
    MisalignmentScenarioSettings,
    TIBs = cms.PSet(
        Dets = cms.PSet(
            dZlocal = cms.double(0.003),
            phiXlocal = cms.double(7e-05),
            dYlocal = cms.double(0.003),
            phiZlocal = cms.double(7e-05),
            dXlocal = cms.double(0.003),
            phiYlocal = cms.double(7e-05)
        ),
        TIBStrings = cms.PSet(
            dZlocal = cms.double(0.003),
            phiXlocal = cms.double(2e-05),
            dYlocal = cms.double(0.003),
            phiZlocal = cms.double(2e-05),
            dXlocal = cms.double(0.003),
            phiYlocal = cms.double(2e-05)
        ),
        distribution = cms.string('gaussian'),
        TIBLayers = cms.PSet(
            dZlocal = cms.double(0.0015),
            phiXlocal = cms.double(1e-05),
            dYlocal = cms.double(0.0015),
            phiZlocal = cms.double(1e-05),
            dXlocal = cms.double(0.0015),
            phiYlocal = cms.double(1e-05)
        ),
        scale = cms.double(1.0)
    ),
    TPBs = cms.PSet(
        Dets = cms.PSet(
            dZlocal = cms.double(0.001),
            phiXlocal = cms.double(4.5e-05),
            dYlocal = cms.double(0.001),
            phiZlocal = cms.double(4.5e-05),
            distribution = cms.string('gaussian'),
            dXlocal = cms.double(0.001),
            phiYlocal = cms.double(4.5e-05)
        ),
        scale = cms.double(1.0),
        TPBLayers = cms.PSet(
            phiXlocal = cms.double(3e-06),
            phiZlocal = cms.double(3e-06),
            dZ = cms.double(0.0005),
            dX = cms.double(0.0005),
            dY = cms.double(0.0005),
            distribution = cms.string('gaussian'),
            phiYlocal = cms.double(3e-06)
        ),
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
    TOBs = cms.PSet(
        scale = cms.double(1.0),
        TOBLayers = cms.PSet(
            dXlocal = cms.double(0.0),
            dZlocal = cms.double(0.0),
            dYlocal = cms.double(0.0)
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
        distribution = cms.string('gaussian'),
        Dets = cms.PSet(
            dZlocal = cms.double(0.0032),
            phiXlocal = cms.double(7e-05),
            dYlocal = cms.double(0.0032),
            phiZlocal = cms.double(7e-05),
            dXlocal = cms.double(0.0032),
            phiYlocal = cms.double(7e-05)
        )
    ),
    TECs = cms.PSet(
        TECDisks = cms.PSet(
            dZlocal = cms.double(0.003),
            phiXlocal = cms.double(5e-06),
            dYlocal = cms.double(0.003),
            phiZlocal = cms.double(5e-06),
            dXlocal = cms.double(0.003),
            phiYlocal = cms.double(5e-06)
        ),
        distribution = cms.string('gaussian'),
        scale = cms.double(1.0),
        Dets = cms.PSet(
            dZlocal = cms.double(0.0022),
            phiXlocal = cms.double(5e-05),
            dYlocal = cms.double(0.0022),
            phiZlocal = cms.double(5e-05),
            dXlocal = cms.double(0.0022),
            phiYlocal = cms.double(5e-05)
        ),
        TECPetals = cms.PSet(
            dZlocal = cms.double(0.0055),
            phiXlocal = cms.double(2e-05),
            dYlocal = cms.double(0.0055),
            phiZlocal = cms.double(2e-05),
            dXlocal = cms.double(0.0055),
            phiYlocal = cms.double(2e-05)
        )
    ),
    TPEs = cms.PSet(
        TPEPanels = cms.PSet(
            dZlocal = cms.double(0.001),
            phiXlocal = cms.double(2.2e-05),
            dYlocal = cms.double(0.001),
            phiZlocal = cms.double(2.2e-05),
            dXlocal = cms.double(0.001),
            phiYlocal = cms.double(2.2e-05)
        ),
        scale = cms.double(1.0),
        TPEHalfDisks = cms.PSet(
            dZlocal = cms.double(0.001),
            phiXlocal = cms.double(1.5e-05),
            dYlocal = cms.double(0.001),
            phiZlocal = cms.double(1.5e-05),
            dXlocal = cms.double(0.001),
            phiYlocal = cms.double(1.5e-05)
        ),
        TPEBlades = cms.PSet(
            dZlocal = cms.double(0.001),
            phiXlocal = cms.double(1.5e-05),
            dYlocal = cms.double(0.001),
            phiZlocal = cms.double(1.5e-05),
            dXlocal = cms.double(0.001),
            phiYlocal = cms.double(1.5e-05)
        ),
        distribution = cms.string('gaussian'),
        Dets = cms.PSet(
            dZlocal = cms.double(0.0005),
            phiXlocal = cms.double(1.1e-05),
            dYlocal = cms.double(0.0005),
            phiZlocal = cms.double(1.1e-05),
            dXlocal = cms.double(0.0005),
            phiYlocal = cms.double(1.1e-05)
        )
    ),
    TIDs = cms.PSet(
        Dets = cms.PSet(
            dZlocal = cms.double(0.005),
            phiXlocal = cms.double(0.00023),
            dYlocal = cms.double(0.005),
            phiZlocal = cms.double(0.00023),
            dXlocal = cms.double(0.005),
            phiYlocal = cms.double(0.00023)
        ),
        TIDDisks = cms.PSet(
            dZlocal = cms.double(0.0025),
            phiXlocal = cms.double(4e-05),
            dYlocal = cms.double(0.0025),
            phiZlocal = cms.double(4e-05),
            dXlocal = cms.double(0.0025),
            phiYlocal = cms.double(4e-05)
        ),
        TIDRings = cms.PSet(
            dZlocal = cms.double(0.005),
            phiXlocal = cms.double(0.00023),
            dYlocal = cms.double(0.005),
            phiZlocal = cms.double(0.00023),
            dXlocal = cms.double(0.005),
            phiYlocal = cms.double(0.00023)
        ),
        distribution = cms.string('gaussian'),
        scale = cms.double(1.0)
    )
)
# -----------------------------------------------------------------------
# 1000 pb-1 misalignment scenario
# See CMS IN 2007/036
Tracker1000pbScenario = cms.PSet(
    MisalignmentScenarioSettings,
    TIBs = cms.PSet(
        Dets = cms.PSet(
            dZlocal = cms.double(0.001),
            phiXlocal = cms.double(2e-05),
            dYlocal = cms.double(0.001),
            phiZlocal = cms.double(2e-05),
            dXlocal = cms.double(0.001),
            phiYlocal = cms.double(2e-05)
        ),
        TIBStrings = cms.PSet(
            dZlocal = cms.double(0.001),
            phiXlocal = cms.double(5e-06),
            dYlocal = cms.double(0.001),
            phiZlocal = cms.double(5e-06),
            dXlocal = cms.double(0.001),
            phiYlocal = cms.double(5e-06)
        ),
        distribution = cms.string('gaussian'),
        TIBLayers = cms.PSet(
            dZlocal = cms.double(0.001),
            phiXlocal = cms.double(5e-06),
            dYlocal = cms.double(0.001),
            phiZlocal = cms.double(5e-06),
            dXlocal = cms.double(0.001),
            phiYlocal = cms.double(5e-06)
        ),
        scale = cms.double(1.0)
    ),
    TPBs = cms.PSet(
        Dets = cms.PSet(
            dZlocal = cms.double(0.0005),
            phiXlocal = cms.double(2.2e-05),
            dYlocal = cms.double(0.0005),
            phiZlocal = cms.double(2.2e-05),
            distribution = cms.string('gaussian'),
            dXlocal = cms.double(0.0005),
            phiYlocal = cms.double(2.2e-05)
        ),
        scale = cms.double(1.0),
        TPBLayers = cms.PSet(
            phiXlocal = cms.double(3e-06),
            phiZlocal = cms.double(3e-06),
            dZ = cms.double(0.0005),
            dX = cms.double(0.0005),
            dY = cms.double(0.0005),
            distribution = cms.string('gaussian'),
            phiYlocal = cms.double(3e-06)
        ),
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
    TOBs = cms.PSet(
        scale = cms.double(1.0),
        TOBLayers = cms.PSet(
            dXlocal = cms.double(0.0),
            dZlocal = cms.double(0.0),
            dYlocal = cms.double(0.0)
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
        distribution = cms.string('gaussian'),
        Dets = cms.PSet(
            dZlocal = cms.double(0.0018),
            phiXlocal = cms.double(4e-05),
            dYlocal = cms.double(0.0018),
            phiZlocal = cms.double(4e-05),
            dXlocal = cms.double(0.0018),
            phiYlocal = cms.double(4e-05)
        )
    ),
    TECs = cms.PSet(
        TECDisks = cms.PSet(
            dZlocal = cms.double(0.002),
            phiXlocal = cms.double(5e-06),
            dYlocal = cms.double(0.002),
            phiZlocal = cms.double(5e-06),
            dXlocal = cms.double(0.002),
            phiYlocal = cms.double(5e-06)
        ),
        distribution = cms.string('gaussian'),
        scale = cms.double(1.0),
        Dets = cms.PSet(
            dZlocal = cms.double(0.0022),
            phiXlocal = cms.double(5e-05),
            dYlocal = cms.double(0.0022),
            phiZlocal = cms.double(5e-05),
            dXlocal = cms.double(0.0022),
            phiYlocal = cms.double(5e-05)
        ),
        TECPetals = cms.PSet(
            dZlocal = cms.double(0.004),
            phiXlocal = cms.double(1.5e-05),
            dYlocal = cms.double(0.004),
            phiZlocal = cms.double(1.5e-05),
            dXlocal = cms.double(0.004),
            phiYlocal = cms.double(1.5e-05)
        )
    ),
    TPEs = cms.PSet(
        TPEPanels = cms.PSet(
            dZlocal = cms.double(0.0005),
            phiXlocal = cms.double(1.1e-05),
            dYlocal = cms.double(0.0005),
            phiZlocal = cms.double(1.1e-05),
            dXlocal = cms.double(0.0005),
            phiYlocal = cms.double(1.1e-05)
        ),
        scale = cms.double(1.0),
        TPEHalfDisks = cms.PSet(
            dZlocal = cms.double(0.0005),
            phiXlocal = cms.double(7e-06),
            dYlocal = cms.double(0.0005),
            phiZlocal = cms.double(7e-06),
            dXlocal = cms.double(0.0005),
            phiYlocal = cms.double(7e-06)
        ),
        TPEBlades = cms.PSet(
            dZlocal = cms.double(0.0005),
            phiXlocal = cms.double(7e-06),
            dYlocal = cms.double(0.0005),
            phiZlocal = cms.double(7e-06),
            dXlocal = cms.double(0.0005),
            phiYlocal = cms.double(7e-06)
        ),
        distribution = cms.string('gaussian'),
        Dets = cms.PSet(
            dZlocal = cms.double(0.0005),
            phiXlocal = cms.double(1.1e-05),
            dYlocal = cms.double(0.0005),
            phiZlocal = cms.double(1.1e-05),
            dXlocal = cms.double(0.0005),
            phiYlocal = cms.double(1.1e-05)
        )
    ),
    TIDs = cms.PSet(
        Dets = cms.PSet(
            dZlocal = cms.double(0.0025),
            phiXlocal = cms.double(0.00011),
            dYlocal = cms.double(0.0025),
            phiZlocal = cms.double(0.00011),
            dXlocal = cms.double(0.0025),
            phiYlocal = cms.double(0.00011)
        ),
        TIDDisks = cms.PSet(
            dZlocal = cms.double(0.0012),
            phiXlocal = cms.double(2e-05),
            dYlocal = cms.double(0.0012),
            phiZlocal = cms.double(2e-05),
            dXlocal = cms.double(0.0012),
            phiYlocal = cms.double(2e-05)
        ),
        TIDRings = cms.PSet(
            dZlocal = cms.double(0.0025),
            phiXlocal = cms.double(0.00011),
            dYlocal = cms.double(0.0025),
            phiZlocal = cms.double(0.00011),
            dXlocal = cms.double(0.0025),
            phiYlocal = cms.double(0.00011)
        ),
        distribution = cms.string('gaussian'),
        scale = cms.double(1.0)
    )
)
# -----------------------------------------------------------------------
# Survey&LAS only misalignment scenario
# See CMS IN 2007/036, table 6, "Updated initial uncertainties"
TrackerSurveyLASOnlyScenario = cms.PSet(
    MisalignmentScenarioSettings,
    TIBs = cms.PSet(
        Dets = cms.PSet(
            dZlocal = cms.double(0.018),
            phiXlocal = cms.double(0.000412),
            dYlocal = cms.double(0.018),
            phiZlocal = cms.double(0.000412),
            dXlocal = cms.double(0.018),
            phiYlocal = cms.double(0.000412)
        ),
        TIBStrings = cms.PSet(
            dZlocal = cms.double(0.045),
            phiXlocal = cms.double(0.000293),
            dYlocal = cms.double(0.045),
            phiZlocal = cms.double(0.000293),
            dXlocal = cms.double(0.045),
            phiYlocal = cms.double(0.000293)
        ),
        distribution = cms.string('gaussian'),
        TIBLayers = cms.PSet(
            dZlocal = cms.double(0.075),
            phiXlocal = cms.double(0.000488),
            dYlocal = cms.double(0.075),
            phiZlocal = cms.double(0.000488),
            dXlocal = cms.double(0.075),
            phiYlocal = cms.double(0.000488)
        ),
        scale = cms.double(1.0)
    ),
    TPBs = cms.PSet(
        Dets = cms.PSet(
            dZlocal = cms.double(0.006),
            phiXlocal = cms.double(0.00027),
            dYlocal = cms.double(0.006),
            phiZlocal = cms.double(0.00027),
            distribution = cms.string('gaussian'),
            dXlocal = cms.double(0.006),
            phiYlocal = cms.double(0.00027)
        ),
        scale = cms.double(1.0),
        TPBLayers = cms.PSet(
            phiXlocal = cms.double(1e-05),
            phiZlocal = cms.double(1e-05),
            dZ = cms.double(0.0337),
            dX = cms.double(0.0225),
            dY = cms.double(0.0225),
            distribution = cms.string('gaussian'),
            phiYlocal = cms.double(1e-05)
        ),
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
    TOBs = cms.PSet(
        scale = cms.double(1.0),
        TOBLayers = cms.PSet(
            dXlocal = cms.double(0.0),
            dZlocal = cms.double(0.0),
            dYlocal = cms.double(0.0)
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
        distribution = cms.string('gaussian'),
        Dets = cms.PSet(
            dZlocal = cms.double(0.0032),
            phiXlocal = cms.double(7.5e-05),
            dYlocal = cms.double(0.0032),
            phiZlocal = cms.double(7.5e-05),
            dXlocal = cms.double(0.0032),
            phiYlocal = cms.double(7.5e-05)
        )
    ),
    TECs = cms.PSet(
        TECDisks = cms.PSet(
            dZlocal = cms.double(0.015),
            phiXlocal = cms.double(1.5e-05),
            dYlocal = cms.double(0.006),
            phiZlocal = cms.double(1.5e-05),
            dXlocal = cms.double(0.006),
            phiYlocal = cms.double(1.5e-05)
        ),
        distribution = cms.string('gaussian'),
        scale = cms.double(1.0),
        Dets = cms.PSet(
            dZlocal = cms.double(0.0022),
            phiXlocal = cms.double(5e-05),
            dYlocal = cms.double(0.0022),
            phiZlocal = cms.double(5e-05),
            dXlocal = cms.double(0.0022),
            phiYlocal = cms.double(5e-05)
        ),
        TECPetals = cms.PSet(
            dZlocal = cms.double(0.007),
            phiXlocal = cms.double(3e-05),
            dYlocal = cms.double(0.007),
            phiZlocal = cms.double(3e-05),
            dXlocal = cms.double(0.007),
            phiYlocal = cms.double(3e-05)
        )
    ),
    TPEs = cms.PSet(
        TPEPanels = cms.PSet(
            dZlocal = cms.double(0.001),
            phiXlocal = cms.double(0.0002),
            dYlocal = cms.double(0.001),
            phiZlocal = cms.double(0.0002),
            dXlocal = cms.double(0.001),
            phiYlocal = cms.double(0.0002)
        ),
        scale = cms.double(1.0),
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
        TPEBlades = cms.PSet(
            dZlocal = cms.double(0.001),
            phiXlocal = cms.double(0.0002),
            dYlocal = cms.double(0.001),
            phiZlocal = cms.double(0.0002),
            dXlocal = cms.double(0.001),
            phiYlocal = cms.double(0.0002)
        ),
        distribution = cms.string('gaussian'),
        Dets = cms.PSet(
            dZlocal = cms.double(0.0005),
            phiXlocal = cms.double(0.0001),
            dYlocal = cms.double(0.0005),
            phiZlocal = cms.double(0.0001),
            dXlocal = cms.double(0.0005),
            phiYlocal = cms.double(0.0001)
        )
    ),
    TIDs = cms.PSet(
        scale = cms.double(1.0),
        TIDEndcaps = cms.PSet(
            dZlocal = cms.double(0.045),
            phiXlocal = cms.double(0.000649),
            dYlocal = cms.double(0.045),
            phiZlocal = cms.double(0.000649),
            dXlocal = cms.double(0.045),
            phiYlocal = cms.double(0.000649)
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
        ),
        Dets = cms.PSet(
            dZlocal = cms.double(0.0054),
            phiXlocal = cms.double(0.00025),
            dYlocal = cms.double(0.0054),
            phiZlocal = cms.double(0.00025),
            dXlocal = cms.double(0.0054),
            phiYlocal = cms.double(0.00025)
        )
    )
)
# -----------------------------------------------------------------------
TrackerSurveyLASCosmicsScenario = cms.PSet(
    MisalignmentScenarioSettings,
    TIBs = cms.PSet(
        Dets = cms.PSet(
            dZlocal = cms.double(0.018),
            phiXlocal = cms.double(0.000412),
            dYlocal = cms.double(0.018),
            phiZlocal = cms.double(0.000412),
            dXlocal = cms.double(0.018),
            phiYlocal = cms.double(0.000412)
        ),
        TIBStrings = cms.PSet(
            dZlocal = cms.double(0.0275),
            phiXlocal = cms.double(0.000179),
            dYlocal = cms.double(0.0275),
            phiZlocal = cms.double(0.000179),
            dXlocal = cms.double(0.0275),
            phiYlocal = cms.double(0.000179)
        ),
        distribution = cms.string('gaussian'),
        TIBLayers = cms.PSet(
            dZlocal = cms.double(0.0425),
            phiXlocal = cms.double(0.000277),
            dYlocal = cms.double(0.0425),
            phiZlocal = cms.double(0.000277),
            dXlocal = cms.double(0.0425),
            phiYlocal = cms.double(0.000277)
        ),
        scale = cms.double(1.0)
    ),
    TPBs = cms.PSet(
        Dets = cms.PSet(
            dZlocal = cms.double(0.006),
            phiXlocal = cms.double(0.00027),
            dYlocal = cms.double(0.006),
            phiZlocal = cms.double(0.00027),
            distribution = cms.string('gaussian'),
            dXlocal = cms.double(0.006),
            phiYlocal = cms.double(0.00027)
        ),
        scale = cms.double(1.0),
        TPBLayers = cms.PSet(
            phiXlocal = cms.double(9e-06),
            phiZlocal = cms.double(9e-06),
            dZ = cms.double(0.0174),
            dX = cms.double(0.0118),
            dY = cms.double(0.0118),
            distribution = cms.string('gaussian'),
            phiYlocal = cms.double(9e-06)
        ),
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
    TOBs = cms.PSet(
        scale = cms.double(1.0),
        TOBLayers = cms.PSet(
            dXlocal = cms.double(0.0),
            dZlocal = cms.double(0.0),
            dYlocal = cms.double(0.0)
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
        distribution = cms.string('gaussian'),
        Dets = cms.PSet(
            dZlocal = cms.double(0.0032),
            phiXlocal = cms.double(7.5e-05),
            dYlocal = cms.double(0.0032),
            phiZlocal = cms.double(7.5e-05),
            dXlocal = cms.double(0.0032),
            phiYlocal = cms.double(7.5e-05)
        )
    ),
    TECs = cms.PSet(
        TECDisks = cms.PSet(
            dZlocal = cms.double(0.015),
            phiXlocal = cms.double(1.5e-05),
            dYlocal = cms.double(0.006),
            phiZlocal = cms.double(1.5e-05),
            dXlocal = cms.double(0.006),
            phiYlocal = cms.double(1.5e-05)
        ),
        distribution = cms.string('gaussian'),
        scale = cms.double(1.0),
        Dets = cms.PSet(
            dZlocal = cms.double(0.0022),
            phiXlocal = cms.double(5e-05),
            dYlocal = cms.double(0.0022),
            phiZlocal = cms.double(5e-05),
            dXlocal = cms.double(0.0022),
            phiYlocal = cms.double(5e-05)
        ),
        TECPetals = cms.PSet(
            dZlocal = cms.double(0.007),
            phiXlocal = cms.double(3e-05),
            dYlocal = cms.double(0.007),
            phiZlocal = cms.double(3e-05),
            dXlocal = cms.double(0.007),
            phiYlocal = cms.double(3e-05)
        )
    ),
    TPEs = cms.PSet(
        TPEPanels = cms.PSet(
            dZlocal = cms.double(0.001),
            phiXlocal = cms.double(0.0002),
            dYlocal = cms.double(0.001),
            phiZlocal = cms.double(0.0002),
            dXlocal = cms.double(0.001),
            phiYlocal = cms.double(0.0002)
        ),
        scale = cms.double(1.0),
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
        TPEBlades = cms.PSet(
            dZlocal = cms.double(0.001),
            phiXlocal = cms.double(0.0002),
            dYlocal = cms.double(0.001),
            phiZlocal = cms.double(0.0002),
            dXlocal = cms.double(0.001),
            phiYlocal = cms.double(0.0002)
        ),
        distribution = cms.string('gaussian'),
        Dets = cms.PSet(
            dZlocal = cms.double(0.0005),
            phiXlocal = cms.double(0.0001),
            dYlocal = cms.double(0.0005),
            phiZlocal = cms.double(0.0001),
            dXlocal = cms.double(0.0005),
            phiYlocal = cms.double(0.0001)
        )
    ),
    TIDs = cms.PSet(
        scale = cms.double(1.0),
        TIDEndcaps = cms.PSet(
            dZlocal = cms.double(0.0225),
            phiXlocal = cms.double(0.000325),
            dYlocal = cms.double(0.0225),
            phiZlocal = cms.double(0.000325),
            dXlocal = cms.double(0.0225),
            phiYlocal = cms.double(0.000325)
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
        ),
        Dets = cms.PSet(
            dZlocal = cms.double(0.0054),
            phiXlocal = cms.double(0.00025),
            dYlocal = cms.double(0.0054),
            phiZlocal = cms.double(0.00025),
            dXlocal = cms.double(0.0054),
            phiYlocal = cms.double(0.00025)
        )
    )
)

