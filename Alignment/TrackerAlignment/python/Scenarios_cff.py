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
# Note: following scenarios are and update for phase1 geometry of phase0 geometry scenarios defined in: 
# https://github.com/cms-sw/cmssw/blob/master/Alignment/TrackerAlignment/python/Scenarios_phase0_cff.py
#
# Updated Hierarchy levels:
# P1PXBLadder
# P1PXBLayer
# P1PXBHalfBarrel
# P1PXBBarrel
# P1PXECPanel
# P1PXECBlade
# P1PXECHalfDisk
# P1PXECHalfCylinder
# P1PXECEndcap
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

#--------------------------------------------------------
# Move ring 7 of of TEC in localY by 1.33 mm

def TECRing7Shift(shift):
  shift_Ring7_Units = cms.PSet( DetUnits = cms.PSet(dYlocal = cms.double(shift)))
  scenario = cms.PSet(
    MisalignmentScenarioSettings,
    TECEndcaps = cms.PSet(
      distribution = cms.string('fixed'),
      TECDisk1_2_3 = cms.PSet(
        TECRing7 = cms.PSet( shift_Ring7_Units )# TECRing7
      ),
      TECDisk4_5_6 = cms.PSet(
        TECRing6 = cms.PSet( shift_Ring7_Units )# TECRing6
      ),
      TECDisk7_8 = cms.PSet(
        TECRing5 = cms.PSet( shift_Ring7_Units )# TECRing5
      ),
      TECDisk9 = cms.PSet(
        TECRing4 = cms.PSet( shift_Ring7_Units )# TECRing4
      )
    )#TECEndcaps
  )#scenario
  return scenario


TECRing7Plus133mmScenario = TECRing7Shift(0.133)
TECRing7Minus133mmScenario = TECRing7Shift(-0.133)

# -----------------------------------------------------------------------
# LS1BPixRepairScenario
# ---------------------
# This scenario mimics the shifts introduced by the BPix repair work
# during LS1. It contains only the aditional movements. It is intended
# for use in realignment studies.
# Object presented in the TkAlignment meeting of Oct 30, 2014
# https://indico.cern.ch/event/337181/, talk by Ekaterina Avdeeva
# -----------------------------------------------------------------------

shift_0006_000027 = cms.PSet(
        dXlocal = cms.double(0.006), phiXlocal = cms.double(0.00027),
        dYlocal = cms.double(0.006), phiYlocal = cms.double(0.00027),
        dZlocal = cms.double(0.006), phiZlocal = cms.double(0.00027)
)# end of shift_0006_000027

LS1BPixRepair_P1PXBHalfBarrel2 = cms.PSet(
    P1PXBHalfBarrel2=cms.PSet(
        P1PXBLayers=cms.PSet(
            dX= cms.double(0.01),
            dY= cms.double(0.01),
            dZ= cms.double(0.02),
            P1PXBLadders=cms.PSet(
                DetUnit1=cms.PSet( shift_0006_000027 ),#DetUnit1
                DetUnit2=cms.PSet( shift_0006_000027 ),#DetUnit2
                DetUnit3=cms.PSet( shift_0006_000027 ),#DetUnit3
                DetUnit4=cms.PSet( shift_0006_000027 ) #DetUnit4
            )#P1PXBLadders
        )#P1PXBLayers
    )#P1PXBHalfBarrel2
)# end of LS1BPixRepair_P1PXBHalfBarrel2

LS1BPixRepairScenario = cms.PSet(
    MisalignmentScenarioSettings,
    LS1BPixRepair_P1PXBHalfBarrel2
)

LS1BPixRepairAndTECRing7Plus133mmScenario = cms.PSet(
    TECRing7Plus133mmScenario,
    LS1BPixRepair_P1PXBHalfBarrel2
)

LS1BPixRepairAndTECRing7Minus133mmScenario = cms.PSet(
    TECRing7Minus133mmScenario,
    LS1BPixRepair_P1PXBHalfBarrel2
)


# -----------------------------------------------------------------------
# CSA14 scenario
# See https://twiki.cern.ch/twiki/bin/viewauth/CMS/TkAl1402 for a
# justification of these values. This scenario has to be applied on top
# of an existing MC object, not on top of IDEAL.
# This scenario has been used to produce the CSA14 50ns scenario
#

shift_0050_10em06 = cms.PSet(
        dXlocal = cms.double(0.0050), phiXlocal = cms.double(10e-06),
        dYlocal = cms.double(0.0050), phiYlocal = cms.double(10e-06),
        dZlocal = cms.double(0.0050), phiZlocal = cms.double(10e-06)
)# end of shift_0050_10em06

TrackerCSA14_P1PXBHalfBarrel1 = cms.PSet(
        P1PXBHalfBarrel1 = cms.PSet(
                P1PXBLayer3 = cms.PSet(
                        P1PXBLadder1 = cms.PSet(
                                DetUnit1 = cms.PSet( shift_0050_10em06 ),
                                DetUnit2 = cms.PSet( shift_0050_10em06 ),
                                DetUnit3 = cms.PSet( shift_0050_10em06 ),
                                DetUnit4 = cms.PSet( shift_0050_10em06 ),
                                DetUnit5 = cms.PSet( shift_0050_10em06 ),
                                DetUnit6 = cms.PSet( shift_0050_10em06 ),
                                DetUnit7 = cms.PSet( shift_0050_10em06 ),
                                DetUnit8 = cms.PSet( shift_0050_10em06 ),
                        ), # P1PXBLadder1
                        P1PXBLadder3 = cms.PSet(
                                DetUnit1 = cms.PSet( shift_0050_10em06 ),
                                DetUnit2 = cms.PSet( shift_0050_10em06 ),
                                DetUnit3 = cms.PSet( shift_0050_10em06 ),
                                DetUnit4 = cms.PSet( shift_0050_10em06 ),
                        ), # P1PXBLadder3
                        P1PXBLadder9 = cms.PSet(
                                DetUnit6 = cms.PSet( shift_0050_10em06 ),
                                DetUnit7 = cms.PSet( shift_0050_10em06 ),
                                DetUnit8 = cms.PSet( shift_0050_10em06 ),
                        ), # P1PXBLadder9
                        P1PXBLadder21 = cms.PSet(
                                DetUnit1 = cms.PSet( shift_0050_10em06 ),
                                DetUnit2 = cms.PSet( shift_0050_10em06 ),
                                DetUnit3 = cms.PSet( shift_0050_10em06 )
                        )# P1PXBLadder21
                )# P1PXBLayer3
        )# P1PXBHalfBarrel1
)#end of TrackerCSA14_P1PXBHalfBarrel1

TrackerCSA14_BarrelsGeneralSettings = cms.PSet(
        distribution = cms.string('gaussian'), scale = cms.double(1.0), scaleError = cms.double(1.0),
        P1PXBHalfBarrels = cms.PSet( distribution = cms.string('flat'), dX = cms.double(0.0005), dY = cms.double(0.0010),
                dZ = cms.double(0.0050), phiX = cms.double(30e-6),
                phiY = cms.double(30e-06), phiZ = cms.double(30e-06)),# P1PXBHalfBarrels
        DetUnits = cms.PSet( dXlocal = cms.double(0.0001), dYlocal = cms.double(0.0001),
                dZlocal = cms.double(0.0001), phiXlocal = cms.double(0.5e-04),
                phiYlocal = cms.double(0.5e-04), phiZlocal = cms.double(0.5e-04))# DetUnits
)#end of TrackerCSA14_BarrelsGeneralSettings

shift_0010_10em06 = cms.PSet(
        dXlocal = cms.double(0.01), phiXlocal = cms.double(10e-06),
        dYlocal = cms.double(0.01), phiYlocal = cms.double(10e-06),
        dZlocal = cms.double(0.01), phiZlocal = cms.double(10e-06)
)# end of shift_001_10em06

shift_00002_05em04 = cms.PSet(
        dXlocal = cms.double(0.0002), phiXlocal = cms.double(0.5e-04),
        dYlocal = cms.double(0.0002), phiYlocal = cms.double(0.5e-04),
        dZlocal = cms.double(0.0002), phiZlocal = cms.double(0.5e-04)
)# end of shift_00002_05em04

TrackerCSA14_OtherThanBarrels = cms.PSet(
 P1PXECEndcap1 = cms.PSet(
    distribution = cms.string('gaussian'), scale = cms.double(1.0), scaleError = cms.double(1.0),
     dX = cms.double(0.01), dY = cms.double(0.01), dZ = cms.double(0.01),
    phiX = cms.double(10e-06), phiY = cms.double(10e-06), phiZ = cms.double(10e-06)
 ),#P1PXECEndcap1
 P1PXECEndcap2 = cms.PSet(
    distribution = cms.string('gaussian'), scale = cms.double(1.0), scaleError = cms.double(1.0),
    P1PXECPanels = cms.PSet ( shift_0010_10em06 ), # P1PXECPanels
    P1PXECHalfCylinder1 = cms.PSet(
        P1PXECHalfDisk1 = cms.PSet(
             P1PXECBlade1 = cms.PSet(
                P1PXECPanel1 = cms.PSet(
                    DetUnits = cms.PSet( shift_0010_10em06 )#DetUnits
                )# P1PXECPanel1
            ), #P1PXECBlade1
             P1PXECBlade8 = cms.PSet(
                P1PXECPanel2 = cms.PSet(
                    DetUnits = cms.PSet( shift_0010_10em06 ) #DetUnits
                )# P1PXECPanel2
            )#P1PXECBlade8
         )#P1PXECHalfDisk1
    ),#P1PXECHalfCylinder1
     P1PXECHalfCylinder2 = cms.PSet(
        P1PXECHalfDisk1 = cms.PSet(
            P1PXECBlade9 = cms.PSet(
                P1PXECPanel2 = cms.PSet(
                    DetUnits = cms.PSet( shift_0010_10em06 ) #DetUnits
                )# P1PXECPanel2
            )#P1PXECBlade9
         )#P1PXECHalfDisk1
    )#P1PXECHalfCylinder2
 ),#P1PXECEndcap2
 TIBHalfBarrels = cms.PSet(distribution = cms.string('gaussian'), scale = cms.double(1.0), scaleError = cms.double(1.0),
     DetUnits = cms.PSet( shift_00002_05em04 )#DetUnits
 ), # TIBHalfBarrels
 TOBHalfBarrels = cms.PSet(distribution = cms.string('gaussian'), scale = cms.double(1.0), scaleError = cms.double(1.0),
     DetUnits = cms.PSet( dXlocal = cms.double(0.0005), dYlocal = cms.double(0.0005),
        dZlocal = cms.double(0.0005), phiXlocal = cms.double(0.5e-04),
        phiYlocal = cms.double(0.5e-04), phiZlocal = cms.double(0.5e-04),)#DetUnits
 ), # TOBHalfBarrels
 TIDEndcaps = cms.PSet(distribution = cms.string('gaussian'), scale = cms.double(1.0), scaleError = cms.double(1.0),
     DetUnits = cms.PSet( shift_00002_05em04 )#DetUnits
 ), # TIDEndcaps
 TECEndcaps = cms.PSet(distribution = cms.string('gaussian'), scale = cms.double(1.0), scaleError = cms.double(1.0),
     DetUnits = cms.PSet( shift_00002_05em04 )#DetUnits
 ) # TECEndcaps
)#end of TrackerCSA14_OtherThanBarrels

TrackerCSA14Scenario = cms.PSet(
        MisalignmentScenarioSettings,
        TrackerCSA14_OtherThanBarrels,
        P1PXBBarrels = cms.PSet(
                TrackerCSA14_BarrelsGeneralSettings,
                TrackerCSA14_P1PXBHalfBarrel1
        )#P1PXBBarrels
)#end of TrackerCSA14Scenario

#------------------------------------------------------------------------
# merged TrackerCSA14 and LS1BPixRepair

TrackerCSA14AndLS1BPixRepairScenario = cms.PSet(
        MisalignmentScenarioSettings,
        TrackerCSA14_OtherThanBarrels,
        P1PXBBarrels = cms.PSet(
                TrackerCSA14_BarrelsGeneralSettings,
                TrackerCSA14_P1PXBHalfBarrel1,
                LS1BPixRepair_P1PXBHalfBarrel2
        )#P1PXBBarrels
)#end of TrackerCSA14AndLS1BPixRepairScenario



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
    P1PXBHalfBarrels = cms.PSet(
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
        P1PXBLayers = cms.PSet(
            phiXlocal = cms.double(7e-06),
            phiZlocal = cms.double(7e-06),
            dZ = cms.double(0.001),
            dX = cms.double(0.001),
            dY = cms.double(0.001),
            distribution = cms.string('gaussian'),
            phiYlocal = cms.double(7e-06)
        ),
        scaleError = cms.double(1.0),
        P1PXBLadders = cms.PSet(
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
    P1PXECEndcaps = cms.PSet(
        P1PXECPanels = cms.PSet(
            dZlocal = cms.double(0.001),
            phiXlocal = cms.double(0.0002),
            dYlocal = cms.double(0.001),
            phiZlocal = cms.double(0.0002),
            dXlocal = cms.double(0.001),
            phiYlocal = cms.double(0.0002)
        ),
        scale = cms.double(1.0),
        scaleError = cms.double(1.0),
        P1PXECHalfDisks = cms.PSet(
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
        P1PXECBlades = cms.PSet(
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
    P1PXBHalfBarrels = cms.PSet(
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
        P1PXBLayers = cms.PSet(
            phiXlocal = cms.double(3e-06),
            phiZlocal = cms.double(3e-06),
            dZ = cms.double(0.0005),
            dX = cms.double(0.0005),
            dY = cms.double(0.0005),
            distribution = cms.string('gaussian'),
            phiYlocal = cms.double(3e-06)
        ),
        scaleError = cms.double(1.0),
        P1PXBLadders = cms.PSet(
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
    P1PXECEndcaps = cms.PSet(
        P1PXECPanels = cms.PSet(
            dZlocal = cms.double(0.001),
            phiXlocal = cms.double(2.2e-05),
            dYlocal = cms.double(0.001),
            phiZlocal = cms.double(2.2e-05),
            dXlocal = cms.double(0.001),
            phiYlocal = cms.double(2.2e-05)
        ),
        scale = cms.double(1.0),
        scaleError = cms.double(1.0),
        P1PXECHalfDisks = cms.PSet(
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
        P1PXECBlades = cms.PSet(
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
    P1PXBHalfBarrels = cms.PSet(
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
        P1PXBLayers = cms.PSet(
            phiXlocal = cms.double(3e-06),
            phiZlocal = cms.double(3e-06),
            dZ = cms.double(0.0005),
            dX = cms.double(0.0005),
            dY = cms.double(0.0005),
            distribution = cms.string('gaussian'),
            phiYlocal = cms.double(3e-06)
        ),
        scaleError = cms.double(1.0),
        P1PXBLadders = cms.PSet(
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
    P1PXECEndcaps = cms.PSet(
        P1PXECPanels = cms.PSet(
            dZlocal = cms.double(0.0005),
            phiXlocal = cms.double(1.1e-05),
            dYlocal = cms.double(0.0005),
            phiZlocal = cms.double(1.1e-05),
            dXlocal = cms.double(0.0005),
            phiYlocal = cms.double(1.1e-05)
        ),
        scale = cms.double(1.0),
        scaleError = cms.double(1.0),
        P1PXECHalfDisks = cms.PSet(
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
        P1PXECBlades = cms.PSet(
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
    P1PXBHalfBarrels = cms.PSet(
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
        P1PXBLayers = cms.PSet(
            phiXlocal = cms.double(1e-05),
            phiZlocal = cms.double(1e-05),
            dZ = cms.double(0.0337),
            dX = cms.double(0.0225),
            dY = cms.double(0.0225),
            distribution = cms.string('gaussian'),
            phiYlocal = cms.double(1e-05)
        ),
        scaleError = cms.double(1.0),
        P1PXBLadders = cms.PSet(
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
    P1PXECEndcaps = cms.PSet(
        P1PXECPanels = cms.PSet(
            dZlocal = cms.double(0.001),
            phiXlocal = cms.double(0.0002),
            dYlocal = cms.double(0.001),
            phiZlocal = cms.double(0.0002),
            dXlocal = cms.double(0.001),
            phiYlocal = cms.double(0.0002)
        ),
        scale = cms.double(1.0),
        scaleError = cms.double(1.0),
        P1PXECHalfDisks = cms.PSet(
            dZlocal = cms.double(0.005),
            phiXlocal = cms.double(0.001),
            dYlocal = cms.double(0.005),
            phiZlocal = cms.double(0.001),
            dXlocal = cms.double(0.005),
            phiYlocal = cms.double(0.001)
        ),
        P1PXECHalfCylinders = cms.PSet(
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
        P1PXECBlades = cms.PSet(
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
    P1PXBHalfBarrels = cms.PSet(
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
        P1PXBLayers = cms.PSet(
            phiXlocal = cms.double(9e-06),
            phiZlocal = cms.double(9e-06),
            dZ = cms.double(0.0174),
            dX = cms.double(0.0118),
            dY = cms.double(0.0118),
            distribution = cms.string('gaussian'),
            phiYlocal = cms.double(9e-06)
        ),
        scaleError = cms.double(1.0),
        P1PXBLadders = cms.PSet(
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
    P1PXECEndcaps = cms.PSet(
        P1PXECPanels = cms.PSet(
            dZlocal = cms.double(0.001),
            phiXlocal = cms.double(0.0002),
            dYlocal = cms.double(0.001),
            phiZlocal = cms.double(0.0002),
            dXlocal = cms.double(0.001),
            phiYlocal = cms.double(0.0002)
        ),
        scale = cms.double(1.0),
        scaleError = cms.double(1.0),
        P1PXECHalfDisks = cms.PSet(
            dZlocal = cms.double(0.005),
            phiXlocal = cms.double(0.001),
            dYlocal = cms.double(0.005),
            phiZlocal = cms.double(0.001),
            dXlocal = cms.double(0.005),
            phiYlocal = cms.double(0.001)
        ),
        P1PXECHalfCylinders = cms.PSet(
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
        P1PXECBlades = cms.PSet(
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
    P1PXBHalfBarrels = cms.PSet(
        P1PXBHalfBarrels = cms.PSet( # FIXME??
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
        P1PXBLadders = cms.PSet(
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
        P1PXBLayers = cms.PSet(
            phiXlocal = cms.double(0.0004),
            phiZlocal = cms.double(0.0004),
            dZ = cms.double(0.02),
            dX = cms.double(0.02),
            dY = cms.double(0.02),
            distribution = cms.string('gaussian'),
            phiYlocal = cms.double(0.0004)
        ),
        P1PXBBarrels = cms.PSet( # FIXME??
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
    P1PXECEndcaps = cms.PSet(
        P1PXECPanels = cms.PSet(
            dZlocal = cms.double(0.005),
            phiXlocal = cms.double(0.002),
            dYlocal = cms.double(0.01),
            phiZlocal = cms.double(0.002),
            dXlocal = cms.double(0.01),
            phiYlocal = cms.double(0.002)
        ),
        scale = cms.double(1.0),
        scaleError = cms.double(1.0),
        P1PXECHalfDisks = cms.PSet(
            dZlocal = cms.double(0.02),
            phiXlocal = cms.double(0.001),
            dYlocal = cms.double(0.02),
            phiZlocal = cms.double(0.001),
            dXlocal = cms.double(0.02),
            phiYlocal = cms.double(0.001)
        ),
        P1PXECEndcaps = cms.PSet( # FIXME??
            dZlocal = cms.double(0.1),
            phiXlocal = cms.double(0.0017),
            dYlocal = cms.double(0.1),
            phiZlocal = cms.double(0.0017),
            dXlocal = cms.double(0.1),
            phiYlocal = cms.double(0.0017)
        ),
        P1PXECHalfCylinders = cms.PSet(
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
        P1PXECBlades = cms.PSet(
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
TrackerCRAFTScenario.P1PXECEndcaps = copy.deepcopy(TrackerSurveyLASOnlyScenario.P1PXECEndcaps)

# -----------------------------------------------------------------------
# Pixel Tracker Only z Shifts and Rotations
# ----------------------------------------------------------------------
PixelTrackerOnlyZShift = cms.PSet(MisalignmentScenarioSettings,
                                  P1PXBHalfBarrel1 = cms.PSet(distribution = cms.string('fixed'),
                                                            DetUnits = cms.PSet(dZ = cms.double(0.0015))
                                                            ),
                                  P1PXBHalfBarrel2 = cms.PSet(distribution = cms.string('fixed'),
                                                            DetUnits = cms.PSet(dZ = cms.double(-0.0015))
                                                            )
                                  )

# -----------------------------------------------------------------------
# Pixel Tracker Fixed Shifts and Rotations
# ----------------------------------------------------------------------
PixelTrackerFixedShiftsAndRotations = cms.PSet(MisalignmentScenarioSettings,
                                          P1PXBHalfBarrel1 = cms.PSet(distribution = cms.string('fixed'),
                                                                    DetUnits = cms.PSet(dX = cms.double(0.0005),
                                                                                        dY = cms.double(0.0010),
                                                                                        dZ = cms.double(0.0015),
                                                                                        phiX = cms.double(30e-6),
                                                                                        phiY = cms.double(30e-06),
                                                                                        phiZ = cms.double(30e-06)
                                                                                        )
                                                                    ),
                                          P1PXBHalfBarrel2 = cms.PSet(distribution = cms.string('fixed'),
                                                                    DetUnits = cms.PSet(dX = cms.double(-0.0005),
                                                                                        dY = cms.double(-0.0010),
                                                                                        dZ = cms.double(-0.0015),
                                                                                        phiX = cms.double(-30e-6),
                                                                                        phiY = cms.double(-30e-06),
                                                                                        phiZ = cms.double(-30e-06)
                                                                                        )
                                                                    )
                                          )


# -----------------------------------------------------------------------
# Pixel Tracker Random Shifts and Rotations
# ----------------------------------------------------------------------

PixelTrackerDicedShiftsAndRotations = cms.PSet(MisalignmentScenarioSettings,
                                               P1PXBBarrels = cms.PSet(distribution = cms.string('gaussian'),
                                                                     scale = cms.double(1.0),
                                                                     scaleError = cms.double(1.0),
                                                                     P1PXBHalfBarrels = cms.PSet(distribution = cms.string('flat'),
                                                                                               dX = cms.double(0.0005),
                                                                                               dY = cms.double(0.0010),
                                                                                               dZ = cms.double(0.0015),
                                                                                               phiX = cms.double(30e-6),
                                                                                               phiY = cms.double(30e-06),
                                                                                               phiZ = cms.double(30e-06)
                                                                                               )
                                                                     )
                                               )



MisalignmentScenario_5mu = MisalignmentScenarioSettings.clone(
  setError = cms.bool(False),
  scale = cms.double(0.01), # shifts in 100um

  P1PXBHalfBarrels = cms.PSet(
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

  P1PXECEndcaps = cms.PSet(
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

MisalignmentScenario_10mu = MisalignmentScenarioSettings.clone(
  setError = cms.bool(False),
  scale = cms.double(0.01), # shifts in 100um

  P1PXBHalfBarrels = cms.PSet(
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

  P1PXECEndcaps = cms.PSet(
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

MisalignmentScenario_15mu = MisalignmentScenarioSettings.clone(
  setError = cms.bool(False),
  scale = cms.double(0.01), # shifts in 100um

  P1PXBHalfBarrels = cms.PSet(
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

  P1PXECEndcaps = cms.PSet(
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

MisalignmentScenario_20mu = MisalignmentScenarioSettings.clone(
  setError = cms.bool(False),
  scale = cms.double(0.01), # shifts in 100um

  P1PXBHalfBarrels = cms.PSet(
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

  P1PXECEndcaps = cms.PSet(
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

MisalignmentScenario_20muGlobalX = MisalignmentScenarioSettings.clone(
  setError = cms.bool(False),
  scale = cms.double(0.01), # shifts in 100um

  P1PXBHalfBarrels = cms.PSet(
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

  P1PXECEndcaps = cms.PSet(
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

MisalignmentScenario_20muLocalX = MisalignmentScenarioSettings.clone(
  setError = cms.bool(False),
  scale = cms.double(0.01), # shifts in 100um

  P1PXBHalfBarrels = cms.PSet(
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

  P1PXECEndcaps = cms.PSet(
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

MisalignmentScenario_20muGlobalY = MisalignmentScenarioSettings.clone(
  scale = cms.double(0.01), # shifts in 100um

  P1PXBHalfBarrels = cms.PSet(
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

  P1PXECEndcaps = cms.PSet(
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

MisalignmentScenario_20muLocalY = MisalignmentScenarioSettings.clone(
  setError = cms.bool(False),
  scale = cms.double(0.01), # shifts in 100um

  P1PXBHalfBarrels = cms.PSet(
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

  P1PXECEndcaps = cms.PSet(
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

MisalignmentScenario_20muGlobalZ = MisalignmentScenarioSettings.clone(
  setError = cms.bool(False),
  scale = cms.double(0.01), # shifts in 100um

  P1PXBHalfBarrels = cms.PSet(
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

  P1PXECEndcaps = cms.PSet(
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

MisalignmentScenario_20muLocalZ = MisalignmentScenarioSettings.clone(
  setError = cms.bool(False),
  scale = cms.double(0.01), # shifts in 100um

  P1PXBHalfBarrels = cms.PSet(
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

  P1PXECEndcaps = cms.PSet(
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

MisalignmentScenario_20muGlobalXY = MisalignmentScenarioSettings.clone(
  scale = cms.double(0.01), # shifts in 100um

  P1PXBHalfBarrels = cms.PSet(
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

  P1PXECEndcaps = cms.PSet(
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

MisalignmentScenario_20muLocalXY = MisalignmentScenarioSettings.clone(
  setError = cms.bool(False),
  scale = cms.double(0.01), # shifts in 100um

  P1PXBHalfBarrels = cms.PSet(
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

  P1PXECEndcaps = cms.PSet(
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

MisalignmentScenario_20muGlobalXYZ = MisalignmentScenarioSettings.clone(
  setError = cms.bool(False),
  scale = cms.double(0.01), # shifts in 100um

  P1PXBHalfBarrels = cms.PSet(
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

  P1PXECEndcaps = cms.PSet(
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

MisalignmentScenario_20muLocalXYZ = MisalignmentScenarioSettings.clone(
  setError = cms.bool(False),
  scale = cms.double(0.01), # shifts in 100um

  P1PXBHalfBarrels = cms.PSet(
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

  P1PXECEndcaps = cms.PSet(
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

MisalignmentScenario_BPIX20muGlobalX = MisalignmentScenarioSettings.clone(
  setError = cms.bool(False),
  scale = cms.double(0.01), # shifts in 100um

  P1PXBHalfBarrels = cms.PSet(
    DetUnits = cms.PSet(
      dX = cms.double(0.2),
    ),
  ),
)

MisalignmentScenario_BPIX20muGlobalY = MisalignmentScenarioSettings.clone(
  setError = cms.bool(False),
  scale = cms.double(0.01), # shifts in 100um

  P1PXBHalfBarrels = cms.PSet(
    DetUnits = cms.PSet(
      dY = cms.double(0.2),
    ),
  ),
)

MisalignmentScenario_BPIX20muGlobalZ = MisalignmentScenarioSettings.clone(
  setError = cms.bool(False),
  scale = cms.double(0.01), # shifts in 100um

  P1PXBHalfBarrels = cms.PSet(
    DetUnits = cms.PSet(
      dZ = cms.double(0.2),
    ),
  ),
)

MisalignmentScenario_BPIX20muLocalX = MisalignmentScenarioSettings.clone(
  setError = cms.bool(False),
  scale = cms.double(0.01), # shifts in 100um

  P1PXBHalfBarrels = cms.PSet(
    DetUnits = cms.PSet(
      dXlocal = cms.double(0.2),
    ),
  ),
)

MisalignmentScenario_BPIX20muLocalY = MisalignmentScenarioSettings.clone(
  setError = cms.bool(False),
  scale = cms.double(0.01), # shifts in 100um

  P1PXBHalfBarrels = cms.PSet(
    DetUnits = cms.PSet(
      dYlocal = cms.double(0.2),
    ),
  ),
)

MisalignmentScenario_BPIX20muLocalZ = MisalignmentScenarioSettings.clone(
  setError = cms.bool(False),
  scale = cms.double(0.01), # shifts in 100um

  P1PXBHalfBarrels = cms.PSet(
    DetUnits = cms.PSet(
      dZlocal = cms.double(0.2),
    ),
  ),
)

MisalignmentScenario_NonMisalignedBPIX = MisalignmentScenarioSettings.clone(
  setError = cms.bool(False),
  scale = cms.double(0.01), # shifts in 100um

  P1PXBHalfBarrels = cms.PSet(
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

  P1PXECEndcaps = cms.PSet(
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

MisalignmentScenarioDifferentSubdetectors = MisalignmentScenarioSettings.clone(
  setError = cms.bool(False),
  scale = cms.double(0.01), # shifts in 100um

  P1PXBHalfBarrels = cms.PSet(
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

  P1PXECEndcaps = cms.PSet(
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

MisalignmentScenarioDifferentSubdetectorsLarge = MisalignmentScenarioSettings.clone(
  setError = cms.bool(False),
  scale = cms.double(0.01), # shifts in 100um

  P1PXBHalfBarrels = cms.PSet(
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

  P1PXECEndcaps = cms.PSet(
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

MisalignmentScenarioDifferentSubdetectorsLocal = MisalignmentScenarioSettings.clone(
  setError = cms.bool(False),
  #~ scale = cms.double(0.01), # shifts in 100um

  P1PXBHalfBarrels = cms.PSet(
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

  P1PXECEndcaps = cms.PSet(
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

MisalignmentScenario10Mu = MisalignmentScenarioSettings.clone(
  setError = cms.bool(False),
  scale = cms.double(0.01), # shifts in 100um

  P1PXBHalfBarrels = cms.PSet(
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

  P1PXECEndcaps = cms.PSet(
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

MisalignmentScenario10MuPhase2 = MisalignmentScenarioSettings.clone(
  setError = cms.bool(False),
  scale = cms.double(0.01), # shifts in 100um

  P2PXBHalfBarrels = cms.PSet(
    DetUnits = cms.PSet(
      dYlocal = cms.double(0.1),
      dXlocal = cms.double(0.1),
    ),
  ),

  P2OTBHalfBarrels = cms.PSet(
    DetUnits = cms.PSet(
      dYlocal = cms.double(0.1),
      dXlocal = cms.double(0.1),
    ),
  ),

  P2PXECEndcaps = cms.PSet(
    DetUnits = cms.PSet(
      dYlocal = cms.double(0.1),
      dXlocal = cms.double(0.1),
    ),
  ),

  P2OTECEndcaps = cms.PSet(
    DetUnits = cms.PSet(
      dYlocal = cms.double(0.1),
      dXlocal = cms.double(0.1),
    ),
  ),

)

MisalignmentScenario10MuPixelPhase2 = MisalignmentScenarioSettings.clone(
  setError = cms.bool(False),
  scale = cms.double(0.01), # shifts in 100um

  P2PXBHalfBarrels = cms.PSet(
    DetUnits = cms.PSet(
      dYlocal = cms.double(0.1),
      dXlocal = cms.double(0.1),
    ),
  ),

  P2PXECEndcaps = cms.PSet(
    DetUnits = cms.PSet(
      dYlocal = cms.double(0.1),
      dXlocal = cms.double(0.1),
    ),
  ),

)

MisalignmentScenario100Mu = MisalignmentScenarioSettings.clone(
  setError = cms.bool(False),
  scale = cms.double(0.01), # shifts in 100um

  P1PXBHalfBarrels = cms.PSet(
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

  P1PXECEndcaps = cms.PSet(
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


MisalignmentScenario200Mu = MisalignmentScenarioSettings.clone(
  setError = cms.bool(False),
  scale = cms.double(0.01),# shifts in 100mum

  P1PXBHalfBarrels = cms.PSet(
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

  P1PXECEndcaps = cms.PSet(
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


MisalignmentScenario300Mu = MisalignmentScenarioSettings.clone(
  setError = cms.bool(False),
  scale = cms.double(0.01),# shifts in 100mum

  P1PXBHalfBarrels = cms.PSet(
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

  P1PXECEndcaps = cms.PSet(
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



MisalignmentScenarioBPIX100Mu = MisalignmentScenarioSettings.clone(
  setError = cms.bool(False),
  scale = cms.double(0.01),# shifts in 100mum

  P1PXBHalfBarrels = cms.PSet(
    DetUnits = cms.PSet(
      dZlocal = cms.double(1),
      dYlocal = cms.double(1),
      dXlocal = cms.double(1),
    ),
  ),
)


MisalignedP1PXB = MisalignmentScenarioSettings.clone(
  setError = cms.bool(False),
  scale = cms.double(0.01), # shifts in 100um

  P1PXBHalfBarrels = cms.PSet(
    DetUnits = cms.PSet(
      dZlocal = cms.double(1),
      dYlocal = cms.double(1),
      dXlocal = cms.double(1),
    )
  )
)

MisalignedP1PXEC = MisalignmentScenarioSettings.clone(
  setError = cms.bool(False),
  scale = cms.double(0.01), # shifts in 100um

  P1PXECEndcaps = cms.PSet(
    DetUnits = cms.PSet(
      dZlocal = cms.double(1),
      dYlocal = cms.double(1),
      dXlocal = cms.double(1),
    )
  )
)

MisalignedTIB = MisalignmentScenarioSettings.clone(
  setError = cms.bool(False),
  scale = cms.double(0.01), # shifts in 100um

  TIBHalfBarrels = cms.PSet(
    DetUnits = cms.PSet(
      dZlocal = cms.double(1),
      dYlocal = cms.double(1),
      dXlocal = cms.double(1),
    )
  )
)

MisalignedTOB = MisalignmentScenarioSettings.clone(
  setError = cms.bool(False),
  scale = cms.double(0.01), # shifts in 100um

  TOBHalfBarrels = cms.PSet(
    DetUnits = cms.PSet(
      dZlocal = cms.double(1),
      dYlocal = cms.double(1),
      dXlocal = cms.double(1),
    )
  )
)

MisalignedTID = MisalignmentScenarioSettings.clone(
  setError = cms.bool(False),
  scale = cms.double(0.01), # shifts in 100um

  TIDEndcaps = cms.PSet(
    DetUnits = cms.PSet(
      dZlocal = cms.double(1),
      dYlocal = cms.double(1),
      dXlocal = cms.double(1),
    )
  )
)

MisalignedTEC = MisalignmentScenarioSettings.clone(
  setError = cms.bool(False),
  scale = cms.double(0.01), # shifts in 100um

  TECEndcaps = cms.PSet(
    DetUnits = cms.PSet(
      dZlocal = cms.double(1),
      dYlocal = cms.double(1),
      dXlocal = cms.double(1),
    )
  )
)


MisalignmentAPEScenarioBase = MisalignmentScenarioSettings.clone(
  # Sigma in mum
  #   BPIX:   20
  #   BPIX-y: 10
  #   FPIX:   10
  #   FPIX-y: 20
  #   TEC:    20
  #   TIB:    10
  #   TID:    10
  #   TOB:    10
  setError = cms.bool(False),
  scale = cms.double(0.0001), # shifts in 1um

  P1PXBHalfBarrels = cms.PSet(
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

  P1PXECEndcaps = cms.PSet(
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

MisalignmentScenario_PhaseI_PseudoAsymptotic = MisalignmentScenarioSettings.clone(
  # This scenarios was created to get a phase-1 scenario with similar
  # performance as the phase-0 asymptotic scenario by applying the RMS values
  # from the phase-0 pixel as gaussian smearing to the phase-1 pixel.
  # The values are documented here:
  # https://twiki.cern.ch/twiki/bin/view/CMS/TkAlPhaseIPreparations#Pseudo_asymptotic_scenario

  # --> Gaussian distribution from 'MisalignmentScenarioSettings'
  setError = cms.bool(False),
  scale = cms.double(0.0001), # shifts in 1um

  P1PXBBarrels = cms.PSet(
    DetUnits = cms.PSet(
      dXlocal = cms.double(0.45),
      dYlocal = cms.double(0.6),
      dZlocal = cms.double(0.8),
      phiXlocal = cms.double(0.07), # 0.07 * 0.0001 = 0.000007 rad -> 0.00040107 degree
      phiYlocal = cms.double(0.05),
      phiZlocal = cms.double(0.5),
    ),
  ),

  P1PXECEndcaps = cms.PSet(
    DetUnits = cms.PSet(
      dXlocal = cms.double(0.8),
      dYlocal = cms.double(1.0),
      dZlocal = cms.double(1.2),
      phiXlocal = cms.double(0.7),
      phiYlocal = cms.double(0.6),
      phiZlocal = cms.double(0.3),
    ),
  ),
)

