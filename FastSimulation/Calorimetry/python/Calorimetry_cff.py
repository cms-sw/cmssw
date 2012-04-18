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
            

            ### changed after tuning - Feb - July - Shilpi Jain
            #RadiusFactor = cms.double(1.096),
            RadiusFactorEB = cms.double(1.096),
            RadiusFactorEE = cms.double(1.25),
            ### changed after tuning - Feb - July - Shilpi Jain
            
            RadiusPreshowerCorrections = cms.vdouble(0.137, 10.3), # default value for maxshower depth dependence-->works fine
            MipsinGeV = cms.vdouble(0.0001421,0.0000812), # increase in mipsinGeV by 75% only in layer1
            #SpotFraction < 0 <=> deactivated. In the case, CoreIntervals and 
            #TailIntervals are used   
            SpotFraction = cms.double(-1.0),
            GapLossProbability = cms.double(0.9),
            SimulatePreshower = cms.bool(True)
        ),
        CalorimeterProperties = cms.PSet(
            # triplet for each p value:  p, k_e(p), k_h(p) ...
            RespCorrP = cms.vdouble(1.0, 1.0, 1.0, 1000.0, 1.0, 1.0),  
            PreshowerLayer2_thickness = cms.double(0.38), # layer2 thickness back to original 
            ECALEndcap_LightCollection = cms.double(0.023),
            PreshowerLayer1_thickness = cms.double(1.65), # increase in thickness of layer 1 by 3%
            PreshowerLayer1_mipsPerGeV = cms.double(17.85),  # 50% decrease in mipsperGeV 
            PreshowerLayer2_mipsPerGeV = cms.double(59.5),
            ECALBarrel_LightCollection = cms.double(0.03),
            HCAL_Sampling = cms.double(0.0035),
            # Watch out ! The following two values are defined wrt the electron shower simulation
            # There are not directly related to the detector properties
            HCAL_PiOverE = cms.double(0.2)
#            HCAL_PiOverE = cms.double(0.4)
        ),
        UnfoldedMode = cms.untracked.bool(False),
        Debug = cms.untracked.bool(False),
#        EvtsToDebug = cms.untracked.vuint32(487),
        HCAL = cms.PSet(
            SimMethod = cms.int32(0), ## 0 - use HDShower, 1 - use HDRShower, 2 - GFLASH
            GridSize = cms.int32(7),
            #-- 0 - simple response, 1 - parametrized response + showering, 2 - tabulated response + showering
            SimOption = cms.int32(2)
        )
    ),
    GFlash = cms.PSet(
      GflashExportToFastSim = cms.bool(True),
      GflashHadronPhysics = cms.string('QGSP_BERT'),
      GflashEMShowerModel = cms.bool(False),
      GflashHadronShowerModel = cms.bool(True),
      GflashHcalOuter = cms.bool(False),
      GflashHistogram = cms.bool(False),
      GflashHistogramName = cms.string('gflash_histogram.root'),
      Verbosity = cms.untracked.int32(0),
      bField = cms.double(3.8),
      watcherOn = cms.bool(False),
      tuning_pList = cms.vdouble()
    )
)

