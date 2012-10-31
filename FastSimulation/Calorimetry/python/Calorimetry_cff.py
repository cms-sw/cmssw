import FWCore.ParameterSet.Config as cms

#Global fast calorimetry parameters
from FastSimulation.Calorimetry.HcalResponse_cfi import *
from FastSimulation.Calorimetry.HSParameters_cfi import *
FamosCalorimetryBlock = cms.PSet(
    Calorimetry = cms.PSet(
        HSParameterBlock,
        HCALResponseBlock,
        ECAL = cms.PSet(
            # See FastSimulation/CaloRecHitsProducer/python/CaloRecHits_cff.py 
            Digitizer = cms.untracked.bool(False),

            # If set to true the simulation in ECAL would be done 1X0 by 1X0
            # this is slow but more adapted to detailed studies.
            # Otherwise roughty 5 steps are used.
            bFixedLength = cms.bool(False),
    
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
            HCAL_PiOverE = cms.double(0.2),
            # HCAL_PiOverE = cms.double(0.4)

            BarrelCalorimeterProperties = cms.PSet(

                 #======  Geometrical material properties ========
    
                 # Light Collection efficiency 
                 lightColl = cms.double(0.03),
                 # Light Collection uniformity
                 lightCollUnif = cms.double(0.003),
                 # Photostatistics (photons/GeV) in the homegeneous material
                 photoStatistics = cms.double(50.E3),
                 # Thickness of the detector in cm
                 thickness = cms.double(23.0),

                 #====== Global parameters of the material ========

                 # Interaction length in cm
                 interactionLength  = cms.double(18.5),
                 Aeff = cms.double(170.87),
                 Zeff = cms.double(68.36),
                 rho = cms.double(8.280),
                 # Radiation length in g/cm^2
                 radLenIngcm2 = cms.double(7.37),

                 # ===== Those parameters might be entered by hand
                 # or calculated out of the previous ones 

                 # Radiation length in cm. If value set to -1, FastSim uses internally the
                 # formula radLenIngcm2/rho
                 radLenIncm = cms.double(0.89), 
                 # Critical energy in GeV. If value set to -1, FastSim uses internally the
                 # formula (2.66E-3*(x0*Z/A)^1.1): 8.74E-3 for ECAL EndCap
                 criticalEnergy = cms.double(8.74E-3),
                 # Moliere Radius in cm.If value set to -1, FastSim uses internally the
                 # formula : Es/criticalEnergy*X0 with Es=sqrt(4*Pi/alphaEM)*me*c^2=0.0212 GeV
                 # This value is known to be 2.190 cm for ECAL Endcap, but the formula gives 2.159 cm
                 moliereRadius = cms.double(2.190),

                 #====== Parameters for sampling ECAL ========

                 # Sampling Fraction: Fs = X0eff/(da+dp) where X0eff is the average X0
                 # of the active and passive media and da/dp their thicknesses
                 Fs = cms.double(0.0),

                 # e/mip for the calorimeter. May be estimated by 1./(1+0.007*(Zp-Za))
                 ehat = cms.double(0.0),

                 # a rough estimate of ECAL resolution sigma/E = resE/sqrt(E)
                 # it is used to generate Nspots in radial profiles.
                 resE = cms.double(1.),

                 # the width in cm of the active layer
                 da = cms.double(0.2),

                 # the width in cm of the passive layer
                 dp = cms.double(0.8),

                 # Is a homogenious detector?
                 bHom = cms.bool(True),

                 # Activate the LogDebug
                 debug = cms.bool(False)

            ),
            
            EndcapCalorimeterProperties = cms.PSet(

                 #======  Geometrical material properties ========
    
                 # Light Collection efficiency 
                 lightColl = cms.double(0.023),
                 # Light Collection uniformity
                 lightCollUnif = cms.double(0.003),
                 # Photostatistics (photons/GeV) in the homegeneous material
                 photoStatistics = cms.double(50.E3),
                 # Thickness of the detector in cm
                 thickness = cms.double(22.0),

                 #====== Global parameters of the material ========

                 # Interaction length in cm
                 interactionLength  = cms.double(18.5),
                 Aeff = cms.double(170.87),
                 Zeff = cms.double(68.36),
                 rho = cms.double(8.280),
                 # Radiation length in g/cm^2
                 radLenIngcm2 = cms.double(7.37),

                 # ===== Those parameters might be entered by hand
                 # or calculated out of the previous ones 

                 # Radiation length in cm. If value set to -1, FastSim uses internally the
                 # formula radLenIngcm2/rho
                 radLenIncm = cms.double(0.89), 
                 # Critical energy in GeV. If value set to -1, FastSim uses internally the
                 # formula (2.66E-3*(x0*Z/A)^1.1): 8.74E-3 for ECAL EndCap
                 criticalEnergy = cms.double(8.74E-3),
                 # Moliere Radius in cm.If value set to -1, FastSim uses internally the
                 # formula : Es/criticalEnergy*X0 with Es=sqrt(4*Pi/alphaEM)*me*c^2=0.0212 GeV
                 # This value is known to be 2.190 cm for ECAL Endcap, but the formula gives 2.159 cm
                 moliereRadius = cms.double(2.190),


                 #====== Parameters for sampling ECAL ========

                 # Sampling Fraction: Fs = X0eff/(da+dp) where X0eff is the average X0
                 # of the active and passive media and da/dp their thicknesses
                 Fs = cms.double(0.0),

                 # e/mip for the calorimeter. May be estimated by 1./(1+0.007*(Zp-Za))
                 ehat = cms.double(0.0),

                 # a rough estimate of ECAL resolution sigma/E = resE/sqrt(E)
                 # it is used to generate Nspots in radial profiles.
                 resE = cms.double(1.),

                 # the width in cm of the active layer
                 da = cms.double(0.2),

                 # the width in cm of the passive layer
                 dp = cms.double(0.8),

                 # Is a homogenious detector?
                 bHom = cms.bool(True),

                 # Activate the LogDebug
                 debug = cms.bool(False)

            )

        ),
        UnfoldedMode = cms.untracked.bool(False),
        Debug = cms.untracked.bool(False),
        useDQM = cms.untracked.bool(False),
#        EvtsToDebug = cms.untracked.vuint32(487),
        HCAL = cms.PSet(
            SimMethod = cms.int32(0), ## 0 - use HDShower, 1 - use HDRShower, 2 - GFLASH
            GridSize = cms.int32(7),
            #-- 0 - simple response, 1 - parametrized response + showering, 2 - tabulated response + showering
            SimOption = cms.int32(2),
            Digitizer = cms.untracked.bool(False),
            samplingHBHE = cms.vdouble(125.44, 125.54, 125.32, 125.13, 124.46,
                                       125.01, 125.22, 125.48, 124.45, 125.90,
                                       125.83, 127.01, 126.82, 129.73, 131.83,
                                       143.52, # HB
                                       210.55, 197.93, 186.12, 189.64, 189.63,
                                       190.28, 189.61, 189.60, 190.12, 191.22,
                                       190.90, 193.06, 188.42, 188.42), #HE
            samplingHF   = cms.vdouble(0.383, 0.368),
            samplingHO   = cms.vdouble(231.0, 231.0, 231.0, 231.0, 360.0, 
                                       360.0, 360.0, 360.0, 360.0, 360.0,
                                       360.0, 360.0, 360.0, 360.0, 360.0),
            smearTimeHF  = cms.untracked.bool(False),
            timeShiftHF  = cms.untracked.double(17.),
            timeSmearingHF  = cms.untracked.double(2.),
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

