import FWCore.ParameterSet.Config as cms

#Global fast calorimetry parameters
from FastSimulation.Calorimetry.HcalResponse_cfi import *
from FastSimulation.Calorimetry.HSParameters_cfi import *
FamosCalorimetryBlock = cms.PSet(
    Calorimetry = cms.PSet(
        HSParameterBlock,
        HCALResponseBlock,
        ECAL = cms.PSet(
            # For the core 10% of the spots for
            CoreIntervals = cms.vdouble(100.0, 0.1),
            # change the radius of the tail of the shower
            RTFactor = cms.double(1.0),
            # change the radius of the core of the shower
            RCFactor = cms.double(1.0),
            # For the tail 10% of r<1RM. 100% otherwise
            TailIntervals = cms.vdouble(1.0, 0.1, 100.0, 1.0),
            FrontLeakageProbability = cms.double(1.0),
            GridSize = cms.int32(7),
            # change globally the Moliere radius 
            RadiusFactor = cms.double(1.096),
            Debug = cms.untracked.bool(False),
            #SpotFraction < 0 <=> deactivated. In the case, CoreIntervals and 
            #TailIntervals are used   
            SpotFraction = cms.double(-1.0),
            GapLossProbability = cms.double(0.9)
        ),
        CalorimeterProperties = cms.PSet(
            PreshowerLayer2_thickness = cms.double(0.38),
            ECALEndcap_LightCollection = cms.double(0.023),
            PreshowerLayer1_thickness = cms.double(1.6),
            PreshowerLayer1_mipsPerGeV = cms.double(35.7),
            PreshowerLayer2_mipsPerGeV = cms.double(59.5),
            ECALBarrel_LightCollection = cms.double(0.03),
            HCAL_Sampling = cms.double(0.0035),
            # Watch out ! The following two values are defined wrt the electron shower simulation
            # There are not directly related to the detector properties
            HCAL_PiOverE = cms.double(0.2)
        ),
        UnfoldedMode = cms.untracked.bool(False),
        HCAL = cms.PSet(
            SimMethod = cms.int32(0), ## 0 - use HDShower, 1 - use HDRShower

            GridSize = cms.int32(7),
            #-- 0 - simple response, 1 - parametrized response + showering, 2 - tabulated response + showering
            SimOption = cms.int32(2)
        )
    )
)

