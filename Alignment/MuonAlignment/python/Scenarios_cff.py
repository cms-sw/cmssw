import FWCore.ParameterSet.Config as cms

#
# This file contains all scenarios as blocks
# A block can be included in a config file as:
#   using <block label>
# in any place where a PSet could be used.
#
# See corresponding .cff files for examples.
# -----------------------------------------------------------------------
# General settings common to all scenarios
MuonMisalignmentScenarioSettings = cms.PSet(
    setRotations = cms.bool(True),
    setTranslations = cms.bool(True),
    seed = cms.int32(1234567),
    distribution = cms.string('gaussian'),
    setError = cms.bool(True),
)
# -----------------------------------------------------------------------
# Example scenario (dummy movements)
ExampleScenario = cms.PSet(
    MuonMisalignmentScenarioSettings,
    DTSectors = cms.PSet(
        scale = cms.double(1.0),
        dZ = cms.double(1.0),
        dX = cms.double(0.0),
        dY = cms.double(0.0),
        distribution = cms.string('gaussian'),
        phiZ = cms.double(0.001),
        phiY = cms.double(0.0),
        phiX = cms.double(0.0)
    )
    #Muon = cms.PSet(
    #    scale = cms.double(1.0),
    #    dZ = cms.double(0.1),
    #    dX = cms.double(0.1),
    #    dY = cms.double(0.2),
    #    distribution = cms.string('gaussian'),
    #    phiZ = cms.double(0.0),
    #    phiY = cms.double(0.0),
    #    phiX = cms.double(0.0)
    #)
)
# -----------------------------------------------------------------------
#  "Misalignment" scenario without misalignment...
MuonNoMovementsScenario = cms.PSet(
    MuonMisalignmentScenarioSettings
)
# -----------------------------------------------------------------------
# Muon 10 inverse pb scenario (as interpreted from AN-2005-036)
# Units: centimeter and radian 
Muon10InversepbScenario = cms.PSet(
    MuonMisalignmentScenarioSettings,
    CSCEndcaps = cms.PSet(
        distribution = cms.string('gaussian'),
        CSCStations = cms.PSet(
            scale = cms.double(1.0),
            scaleError = cms.double(1.0),
            dZ = cms.double(0.5),
            dX = cms.double(0.2),
            dY = cms.double(0.2),
            CSCChambers = cms.PSet(
                dZ = cms.double(0.05),
                dX = cms.double(0.05),
                dY = cms.double(0.05),
                phiZ = cms.double(0.00025),
                phiY = cms.double(0.00025),
                phiX = cms.double(0.00025)
            ),
            phiZ = cms.double(0.00025),
            phiY = cms.double(0.00025),
            phiX = cms.double(0.00025)
        )
    ),
    DTBarrels = cms.PSet(
        distribution = cms.string('gaussian'),
        DTWheels = cms.PSet(
            scale = cms.double(1.0),
            scaleError = cms.double(1.0),
            DTChambers = cms.PSet(
                dZ = cms.double(0.05),
                dX = cms.double(0.05),
                dY = cms.double(0.05),
                phiZ = cms.double(0.00025),
                phiY = cms.double(0.00025),
                phiX = cms.double(0.00025)
            ),
            dZ = cms.double(0.3),
            dX = cms.double(0.25),
            dY = cms.double(0.25),
            phiZ = cms.double(0.00025),
            phiY = cms.double(0.00025),
            phiX = cms.double(0.00025)
        )
    )
)
# -----------------------------------------------------------------------
# Muon 100 inverse pb scenario (as interpreted from AN-2005-036)
#
# Units: centimeter and radian
#
Muon100InversepbScenario = cms.PSet(
    MuonMisalignmentScenarioSettings,
    CSCSectors = cms.PSet(
        scale = cms.double(1.0),
        scaleError = cms.double(1.0),
        dZ = cms.double(0.1),
        dX = cms.double(0.1),
        dY = cms.double(0.1),
        phiZ = cms.double(0.00025),
        phiY = cms.double(0.00025),
        phiX = cms.double(0.00025)
    ),
    CSCEndcaps = cms.PSet(
        CSCChambers = cms.PSet(
            scale = cms.double(1.0),
            scaleError = cms.double(1.0),
            dZ = cms.double(0.02),
            dX = cms.double(0.02),
            dY = cms.double(0.02),
            phiZ = cms.double(0.0001),
            phiY = cms.double(0.0001),
            phiX = cms.double(0.0001)
        )
    ),
    DTSectors = cms.PSet(
        scale = cms.double(1.0),
        scaleError = cms.double(1.0),
        dZ = cms.double(0.1),
        dX = cms.double(0.1),
        dY = cms.double(0.1),
        phiZ = cms.double(0.00025),
        phiY = cms.double(0.00025),
        phiX = cms.double(0.00025)
    ),
    DTBarrels = cms.PSet(
        DTChambers = cms.PSet(
            scale = cms.double(1.0),
            scaleError = cms.double(1.0),
            dZ = cms.double(0.02),
            dX = cms.double(0.02),
            dY = cms.double(0.02),
            phiZ = cms.double(0.0001),
            phiY = cms.double(0.0001),
            phiX = cms.double(0.0001)
        )
    ),
    Muon = cms.PSet(
        scale = cms.double(1.0),
        scaleError = cms.double(1.0),
        dZ = cms.double(0.1),
        dX = cms.double(0.1),
        dY = cms.double(0.1),
        distribution = cms.string('gaussian'),
        phiZ = cms.double(0.00025),
        phiY = cms.double(0.00025),
        phiX = cms.double(0.00025)
    )
)
#----------------------------------------------//
#----- New 0 inverse pb scenario for 2008 -----//
#----------------------------------------------//
Muon0inversePbScenario2008 = cms.PSet(
    MuonMisalignmentScenarioSettings,
    DTSectors = cms.PSet(
        scale = cms.double(1.0),
        scaleError = cms.double(1.0),
        dZ = cms.double(0.1),
        dX = cms.double(0.05),
        dY = cms.double(0.05),
        phiZ = cms.double(0.0001),
        phiY = cms.double(5e-05),
        phiX = cms.double(5e-05)
    ),
    CSCEndcaps = cms.PSet(
        distribution = cms.string('gaussian'),
        CSCStations = cms.PSet(
            scale = cms.double(1.0),
            scaleError = cms.double(1.0),
            CSCRings = cms.PSet(
                scale = cms.double(1.0),
                scaleError = cms.double(1.0),
                dZ = cms.double(0.2),
                dX = cms.double(0.04),
                dY = cms.double(0.04),
                CSCChambers = cms.PSet(
                    scale = cms.double(1.0),
                    dZlocal = cms.double(0.15),
                    phiXlocal = cms.double(0.001),
                    scaleError = cms.double(1.0),
                    dYlocal = cms.double(0.1),
                    phiZlocal = cms.double(0.0007),
                    dXlocal = cms.double(0.07),
                    phiYlocal = cms.double(0.001)
                ),
                phiZ = cms.double(0.0001),
                phiY = cms.double(0.0004),
                phiX = cms.double(0.0004)
            ),
            dZ = cms.double(0.2),
            dX = cms.double(0.15),
            dY = cms.double(0.15),
            phiZ = cms.double(0.0002),
            phiY = cms.double(0.0003),
            phiX = cms.double(0.0003)
        )
    ),
    DTBarrels = cms.PSet(
        distribution = cms.string('gaussian'),
        DTWheels = cms.PSet(
            scale = cms.double(1.0),
            scaleError = cms.double(1.0),
            DTChambers = cms.PSet(
                scale = cms.double(1.0),
                dZlocal = cms.double(0.1),
                phiXlocal = cms.double(0.0007),
                scaleError = cms.double(1.0),
                dYlocal = cms.double(0.1),
                phiZlocal = cms.double(0.0005),
                dXlocal = cms.double(0.07),
                phiYlocal = cms.double(0.0005)
            ),
            dZ = cms.double(0.2),
            dX = cms.double(0.15),
            dY = cms.double(0.15),
            phiZ = cms.double(0.0002),
            phiY = cms.double(0.0003),
            phiX = cms.double(0.0003)
        )
    ),
    CSCSectors = cms.PSet(
        scale = cms.double(1.0),
        scaleError = cms.double(1.0),
        dZ = cms.double(0.05),
        dX = cms.double(0.05),
        dY = cms.double(0.05),
        phiZ = cms.double(5e-05),
        phiY = cms.double(5e-05),
        phiX = cms.double(5e-05)
    )
)
#------- End of 0 inverse pb scenario 2008 ------//
#---------------------------------------------//
#---- New 10 inverse pb scenario for 2008 ----//
#---------------------------------------------//
Muon10inversePbScenario2008 = cms.PSet(
    MuonMisalignmentScenarioSettings,
    DTSectors = cms.PSet(
        scale = cms.double(1.0),
        scaleError = cms.double(1.0),
        dZ = cms.double(0.1),
        dX = cms.double(0.05),
        dY = cms.double(0.05),
        phiZ = cms.double(0.0001),
        phiY = cms.double(5e-05),
        phiX = cms.double(5e-05)
    ),
    CSCEndcaps = cms.PSet(
        distribution = cms.string('gaussian'),
        CSCStations = cms.PSet(
            scale = cms.double(1.0),
            scaleError = cms.double(1.0),
            CSCRings = cms.PSet(
                scale = cms.double(1.0),
                scaleError = cms.double(1.0),
                dZ = cms.double(0.04),
                dX = cms.double(0.04),
                dY = cms.double(0.04),
                CSCChambers = cms.PSet(
                    scale = cms.double(1.0),
                    dZlocal = cms.double(0.15),
                    phiXlocal = cms.double(0.0007),
                    scaleError = cms.double(1.0),
                    dYlocal = cms.double(0.07),
                    phiZlocal = cms.double(0.0005),
                    dXlocal = cms.double(0.05),
                    phiYlocal = cms.double(0.0007)
                ),
                phiZ = cms.double(0.0001),
                phiY = cms.double(0.0001),
                phiX = cms.double(0.0001)
            ),
            dZ = cms.double(0.2),
            dX = cms.double(0.07),
            dY = cms.double(0.07),
            phiZ = cms.double(0.0001),
            phiY = cms.double(0.0003),
            phiX = cms.double(0.0003)
        )
    ),
    DTBarrels = cms.PSet(
        distribution = cms.string('gaussian'),
        DTWheels = cms.PSet(
            scale = cms.double(1.0),
            scaleError = cms.double(1.0),
            DTChambers = cms.PSet(
                scale = cms.double(1.0),
                dZlocal = cms.double(0.1),
                phiXlocal = cms.double(0.0007),
                scaleError = cms.double(1.0),
                dYlocal = cms.double(0.1),
                phiZlocal = cms.double(0.0003),
                dXlocal = cms.double(0.05),
                phiYlocal = cms.double(0.0003)
            ),
            dZ = cms.double(0.1),
            dX = cms.double(0.07),
            dY = cms.double(0.07),
            phiZ = cms.double(0.0001),
            phiY = cms.double(0.00015),
            phiX = cms.double(0.00015)
        )
    ),
    CSCSectors = cms.PSet(
        scale = cms.double(1.0),
        scaleError = cms.double(1.0),
        dZ = cms.double(0.05),
        dX = cms.double(0.05),
        dY = cms.double(0.05),
        phiZ = cms.double(5e-05),
        phiY = cms.double(5e-05),
        phiX = cms.double(5e-05)
    )
)
#------- End of 10 inverse pb scenario 2008 ------//
#-------------------------------------------//
#---New 100 inverse pb scenario for 2008----//
#-------------------------------------------//
Muon100inversePbScenario2008 = cms.PSet(
    MuonMisalignmentScenarioSettings,
    CSCEndcaps = cms.PSet(
        DTSectors = cms.PSet(
            scale = cms.double(1.0),
            scaleError = cms.double(1.0),
            dZ = cms.double(0.1),
            dX = cms.double(0.05),
            dY = cms.double(0.05),
            phiZ = cms.double(0.0001),
            phiY = cms.double(5e-05),
            phiX = cms.double(5e-05)
        ),
        CSCSectors = cms.PSet(
            scale = cms.double(1.0),
            scaleError = cms.double(1.0),
            dZ = cms.double(0.05),
            dX = cms.double(0.05),
            dY = cms.double(0.05),
            phiZ = cms.double(5e-05),
            phiY = cms.double(5e-05),
            phiX = cms.double(5e-05)
        ),
        distribution = cms.string('gaussian'),
        CSCRings = cms.PSet(
            scale = cms.double(1.0),
            scaleError = cms.double(1.0),
            dZ = cms.double(0.3),
            dX = cms.double(0.02),
            dY = cms.double(0.05),
            CSCChambers = cms.PSet(
                scale = cms.double(1.0),
                dZlocal = cms.double(0.3),
                phiXlocal = cms.double(0.0005),
                scaleError = cms.double(1.0),
                dYlocal = cms.double(0.15),
                phiZlocal = cms.double(0.0001),
                dXlocal = cms.double(0.05),
                phiYlocal = cms.double(0.0003)
            ),
            phiZ = cms.double(0.0002),
            phiY = cms.double(0.0005),
            phiX = cms.double(0.001)
        )
    ),
    DTBarrels = cms.PSet(
        distribution = cms.string('gaussian'),
        DTWheels = cms.PSet(
            scale = cms.double(1.0),
            scaleError = cms.double(1.0),
            DTChambers = cms.PSet(
                scale = cms.double(1.0),
                dZlocal = cms.double(0.1),
                phiXlocal = cms.double(5e-05),
                scaleError = cms.double(1.0),
                dYlocal = cms.double(0.07),
                phiZlocal = cms.double(7e-05),
                dXlocal = cms.double(0.05),
                phiYlocal = cms.double(7e-05)
            ),
            dZ = cms.double(0.05),
            dX = cms.double(0.03),
            dY = cms.double(0.03),
            phiZ = cms.double(0.00015),
            phiY = cms.double(0.0002),
            phiX = cms.double(0.0002)
        )
    )
)
#------- End of 100 inverse pb scenario 2008 ------//
# -----------------------------------------------------------------------
# Survey Only misalignment scenario (as interpreted from AN-2005-036)
#
# Units: centimeter and radian 
#
MuonSurveyOnlyScenario = cms.PSet(
    MuonMisalignmentScenarioSettings,
    CSCEndcaps = cms.PSet(
        scale = cms.double(1.0),
        dZ = cms.double(0.25),
        dX = cms.double(0.25),
        dY = cms.double(0.25),
        distribution = cms.string('gaussian'),
        CSCChambers = cms.PSet(
            dZ = cms.double(0.1),
            phiZ = cms.double(0.0005),
            dX = cms.double(0.1),
            dY = cms.double(0.1)
        ),
        phiZ = cms.double(0.00025)
    ),
    DTBarrels = cms.PSet(
        scale = cms.double(1.0),
        DTChambers = cms.PSet(
            dZ = cms.double(0.1),
            phiZ = cms.double(0.0005),
            dX = cms.double(0.1),
            dY = cms.double(0.1)
        ),
        dZ = cms.double(0.25),
        dX = cms.double(0.25),
        dY = cms.double(0.25),
        distribution = cms.string('gaussian'),
        phiZ = cms.double(0.00025)
    )
)
# -----------------------------------------------------------------------
# Muon Short Term misalignment scenario (as interpreted from AN-2005-036)
#
# Units: centimeter and radian 
#
MuonShortTermScenario = cms.PSet(
    MuonMisalignmentScenarioSettings,
    CSCEndcaps = cms.PSet(
        scale = cms.double(1.0),
        dZ = cms.double(0.1),
        dX = cms.double(0.1),
        dY = cms.double(0.1),
        distribution = cms.string('gaussian'),
        CSCChambers = cms.PSet(
            dZ = cms.double(0.1),
            phiZ = cms.double(0.0005),
            dX = cms.double(0.1),
            dY = cms.double(0.1)
        ),
        phiZ = cms.double(0.0002)
    ),
    DTBarrels = cms.PSet(
        scale = cms.double(1.0),
        DTChambers = cms.PSet(
            dZ = cms.double(0.1),
            phiZ = cms.double(0.00025),
            dX = cms.double(0.1),
            dY = cms.double(0.1)
        ),
        dZ = cms.double(0.1),
        dX = cms.double(0.1),
        dY = cms.double(0.1),
        distribution = cms.string('gaussian'),
        phiZ = cms.double(0.0002)
    )
)
# -----------------------------------------------------------------------
# Muon Long Term misalignment scenario (as interpreted from AN-2005-036)
#
# Units: centimeter and radian 
#
MuonLongTermScenario = cms.PSet(
    MuonMisalignmentScenarioSettings,
    CSCEndcaps = cms.PSet(
        scale = cms.double(1.0),
        dZ = cms.double(0.02),
        dX = cms.double(0.02),
        dY = cms.double(0.02),
        distribution = cms.string('gaussian'),
        CSCChambers = cms.PSet(
            dZ = cms.double(0.04),
            phiZ = cms.double(0.0001),
            dX = cms.double(0.02),
            dY = cms.double(0.02)
        ),
        phiZ = cms.double(4e-05)
    ),
    DTBarrels = cms.PSet(
        scale = cms.double(1.0),
        DTChambers = cms.PSet(
            dZ = cms.double(0.02),
            phiZ = cms.double(5e-05),
            dX = cms.double(0.02),
            dY = cms.double(0.02)
        ),
        dZ = cms.double(0.02),
        dX = cms.double(0.02),
        dY = cms.double(0.02),
        distribution = cms.string('gaussian'),
        phiZ = cms.double(4e-05)
    )
)

