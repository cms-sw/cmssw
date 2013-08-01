import FWCore.ParameterSet.Config as cms

# Material effects to be simulated in the tracker material and associated cuts
TrackerMaterialBlock = cms.PSet(
    TrackerMaterial = cms.PSet(

        # version 0 = Tracker geometry used between CMSSW_1_2_0 and CMSSW_1_4_10. Works for CSA07; 
        # version 1 = Tuned to CMSSW_1_7_0 geometry
        # version 2 = Tuned to CMSSW_1_8_0 geometry
        # version 3 = Tuned to CMSSW_3_0_0 geometry
        # version 4 = Tuned to CMSSW_3_1_2 and CMSSW_3_3_0 geometries 
        TrackerMaterialVersion = cms.uint32(4),

        #**********************************************************************
        # Thickness of all (sensitive and dead) layers in x/X0
        #**********************************************************************
        # Beam Pipe
        BeamPipeThickness = cms.vdouble(0.0038, 0.00265, 0.00265, 0.00265, 0.00240 ),
        # Pixel Barrel Layers 1-3
        PXBThickness = cms.vdouble(0.0222, 0.0217, 0.0217, 0.0217, 0.017134), 
        # Pixel Barrel services at the end of layers 1-3
        PXB1CablesThickness = cms.vdouble(0.1, 0.042, 0.042, 0.000, 0.000), 
        PXB2CablesThickness = cms.vdouble(0.04, 0.042, 0.042, 0.000, 0.000),
        PXB3CablesThickness = cms.vdouble(0.03, 0.042, 0.042, 0.000, 0.000),
        # Pixel Barrel outside cables
        PXBOutCables1Thickness = cms.vdouble(0.04, 0.04, 0.04, 0.04, 0.04), 
        PXBOutCables2Thickness = cms.vdouble(0.025, 0.015, 0.015, 0.012, 0.012),
        # Pixel Disks 1-2
        PXDThickness = cms.vdouble(0.044, 0.058, 0.058, 0.058, 0.02055),
        # Pixel Endcap outside cables
        PXDOutCables1Thickness = cms.vdouble(0.023, 0.034, 0.034, 0.050, 0.058431),
        PXDOutCables2Thickness = cms.vdouble(0.085, 0.185, 0.250, 0.250, 0.265),
        # Tracker Inner barrel layers 1-4
        TIBLayer1Thickness = cms.vdouble(0.06, 0.053, 0.053, 0.053, 0.053),
        TIBLayer2Thickness = cms.vdouble(0.047, 0.053, 0.053, 0.053, 0.053),
        TIBLayer3Thickness = cms.vdouble(0.035, 0.035, 0.035, 0.035, 0.035),
        TIBLayer4Thickness = cms.vdouble(0.033, 0.04, 0.04, 0.04, 0.04),
        # TIB outside services (endcap)
        TIBOutCables1Thickness = cms.vdouble(0.04, 0.108, 0.108, 0.130, 0.130),
        TIBOutCables2Thickness = cms.vdouble(0.04, 0.0, 0.0, 0.0, 0.0),
        # Tracker Inner disks layers 1-3
        TIDLayer1Thickness = cms.vdouble(0.05, 0.04, 0.04, 0.04, 0.04),
        TIDLayer2Thickness = cms.vdouble(0.05, 0.04, 0.04, 0.04, 0.04),
        TIDLayer3Thickness = cms.vdouble(0.05, 0.055, 0.055, 0.055, 0.055),
        # TID outside wall (endcap)
        TIDOutsideThickness = cms.vdouble(0.07, 0.074, 0.074, 0.074, 0.074),
        # TOB inside wall (barrel)
        TOBInsideThickness = cms.vdouble(0.017, 0.009, 0.009, 0.009, 0.009),
        # Tracker Outer barrel layers 1-6
        TOBLayer1Thickness = cms.vdouble(0.044, 0.03, 0.03, 0.03, 0.03),
        TOBLayer2Thickness = cms.vdouble(0.044, 0.03, 0.03, 0.03, 0.03),
        TOBLayer3Thickness = cms.vdouble(0.033, 0.022, 0.022, 0.022, 0.022),
        TOBLayer4Thickness = cms.vdouble(0.033, 0.022, 0.022, 0.022, 0.022),
        TOBLayer5Thickness = cms.vdouble(0.033, 0.022, 0.022, 0.022, 0.022),
        TOBLayer6Thickness = cms.vdouble(0.033, 0.022, 0.022, 0.022, 0.022),
        # TOB services (endcap)
        TOBOutsideThickness = cms.vdouble(0.09, 0.15, 0.15, 0.15, 0.15),
        # Tracker EndCap disks layers 1-9
        TECLayerThickness = cms.vdouble(0.041, 0.045, 0.045, 0.050, 0.050),
        # TOB outside wall (barrel)
        BarrelCablesThickness = cms.vdouble(0.1, 0.038, 0.038, 0.042, 0.042),
        # TEC outside wall (endcap)
        EndcapCables1Thickness = cms.vdouble(0.26, 0.21, 0.21, 0.21, 0.21),
        EndcapCables2Thickness = cms.vdouble(0.08, 0.0, 0.0, 0.0, 0.0),

        #**********************************************************************
        # Position of dead material layers (cables, services, etc.)
        #**********************************************************************
        # Beam pipe
        BeamPipeRadius = cms.vdouble(3.0, 3.0, 3.0, 3.0, 2.44),
        BeamPipeLength = cms.vdouble(26.4, 28.3, 28.3, 28.3, 28.3),
        # Cables and Services at the end of PIXB1,2,3 ("disk")
        PXB1CablesInnerRadius = cms.vdouble(3.6, 3.7, 3.7, 3.7, 3.7),
        PXB2CablesInnerRadius = cms.vdouble(6.1, 6.3, 6.3, 6.3, 6.3),
        PXB3CablesInnerRadius = cms.vdouble(8.5, 9.0, 9.0, 9.0, 9.0),
        # Pixel Barrel Outside walls and cables (endcap)
        PXBOutCables1InnerRadius = cms.vdouble(11.9, 11.9, 11.9, 4.2, 4.2),
        PXBOutCables1OuterRadius = cms.vdouble(15.5, 15.5, 15.5, 16.5, 16.5),
        PXBOutCables1ZPosition = cms.vdouble(27.999, 28.799, 28.799, 28.799, 28.799),
        PXBOutCables2InnerRadius = cms.vdouble(3.8, 3.8, 3.8, 3.8, 3.8),
        PXBOutCables2OuterRadius = cms.vdouble(16.5, 16.5, 16.5, 16.5, 16.5),
        PXBOutCables2ZPosition = cms.vdouble(28.0, 28.8, 28.8, 28.8, 28.8),
        # Pixel Outside walls and cables (barrel and endcap)
        PixelOutCablesRadius = cms.vdouble(17.1, 17.5, 17.5, 17.5, 19.5), ## was 17.5
        PixelOutCablesLength = cms.vdouble(64.8, 72.0, 72.0, 65.0, 65.0), ## was 65.0
        PixelOutCablesInnerRadius = cms.vdouble(3.0, 3.0, 7.197, 7.2, 6.5),
        PixelOutCablesOuterRadius = cms.vdouble(17.3, 17.61, 17.61, 17.61, 19.61), ## was 17.61
        PixelOutCablesZPosition = cms.vdouble(64.9, 72.1, 72.1, 65.1, 65.1),      ## was 65.1
        # Tracker Inner Barrel Outside Cables and walls (endcap)
        TIBOutCables1InnerRadius = cms.vdouble(22.5, 22.5, 22.5, 22.5, 22.5),
        TIBOutCables1OuterRadius = cms.vdouble(53.9, 53.9, 53.9, 53.9, 53.9),
        TIBOutCables1ZPosition = cms.vdouble(75.001, 74.0, 74.0, 74.0, 74.0), 
        TIBOutCables2OuterRadius = cms.vdouble(53.901, 53.901, 53.901, 53.901, 53.901),
        TIBOutCables2InnerRadius = cms.vdouble(35.5, 35.5, 35.5, 35.5, 35.5),
        TIBOutCables2ZPosition = cms.vdouble(75.001, 74.001, 74.001, 74.001, 74.001),
        # Tracker Inner Disks Outside Cables and walls (endcap)
        TIDOutCablesInnerRadius = cms.vdouble(32.0, 22.0, 22.0, 22.0, 22.0), 
        TIDOutCablesZPosition = cms.vdouble(108.0, 108.0, 108.0, 108.0, 108.0),
        # Tracker outer barrel Inside wall (barrel)
        TOBInCablesRadius = cms.vdouble(54.5, 54.5, 54.6, 54.6, 54.6),
        TOBInCablesLength = cms.vdouble(108.2, 108.2, 108.2, 108.2, 108.2),
        # Tracker Outer Barrel Outside Cables and walls (barrel and endcap)
        TOBOutCablesInnerRadius = cms.vdouble(55.0, 55.0, 55.0, 55.0, 55.0),
        TOBOutCablesOuterRadius = cms.vdouble(109.5, 111.0, 111.0, 111.0, 111.0),
        TOBOutCablesZPosition = cms.vdouble(110.0, 115.0, 115.0, 115.0, 115.0),
        TOBOutCablesRadius = cms.vdouble(119.5, 119.5, 119.5, 119.5, 119.5),
        TOBOutCablesLength = cms.vdouble(299.9, 299.9, 299.9, 299.9, 299.9),
        # Tracker Endcaps Outside Cables and walls (endcaps)
        TECOutCables1InnerRadius = cms.vdouble(6.0, 30.0, 4.42, 4.42, 4.42), 
        TECOutCables1OuterRadius = cms.vdouble(120.001, 120.001, 120.001, 120.001, 120.001),
        TECOutCables1ZPosition = cms.vdouble(300.0, 300.0, 300.0, 300.0, 300.0),
        TECOutCables2InnerRadius = cms.vdouble(70.0, 68.0, 68.0, 68.0, 68.0), 
        TECOutCables2OuterRadius = cms.vdouble(120.001, 120.001, 120.001, 120.001, 120.001),
        TECOutCables2ZPosition = cms.vdouble(300.0, 300.0, 300.0, 300.0, 300.0),


        #*********************************************************************
        # Fudge factor to reproduce the layer inhomgeneties
        # Tuned to 170 only.
        #*********************************************************************
        # Layer number (hard-coded in TrackerInteractionGeometry.cc)
        # Convention is < 100 for sensitive layer, > 100 for dead material
        # (although this convention is used nowhere...)
        FudgeLayer = cms.vuint32(
        104,  # Pixel Barrel services
        104,  # Pixel Barrel services
        104,  # Pixel Barrel services
        104,  # Pixel Barrel services
        104,  # Pixel Barrel services
        104,  # Pixel Barrel services
        104,  # Pixel Barrel services
	106,  # Pixel wall 
	106,  # Pixel wall 
	107,  # Pixel endcap services
	107,  # Pixel endcap services
	107,  # Pixel endcap services
	6,    # TIB1 services  
	7,    # TIB2 services  
	8,    # TIB3 services  
	9,    # TIB4 services  
	10,   # TID Layer 1
	11,   # TID Layer 2
	12,   # TID Layer 3
	110,  # TID outside services
	110,  # TID outside services
	111,  # TOB inside services
	111,  # TOB inside services
	13,   # TOB Layer1
	13,   # TOB Layer1
	13,   # TOB Layer1
	13,   # TOB Layer1
	14,   # TOB Layer2
	14,   # TOB Layer2
	14,   # TOB Layer2
	14,   # TOB Layer2
	15,   # TOB Layer3
	15,   # TOB Layer3
	15,   # TOB Layer3
	15,   # TOB Layer3
	16,   # TOB Layer4
	16,   # TOB Layer4
	16,   # TOB Layer4
	16,   # TOB Layer4
	17,   # TOB Layer5
	17,   # TOB Layer5
	17,   # TOB Layer5
	17,   # TOB Layer5
	18,   # TOB Layer6
	18,   # TOB Layer6
	18,   # TOB Layer6
	18,   # TOB Layer6
	112,  # TOB services
	112,  # TOB services
	112,  # TOB services
	 19,  # TEC Layer 1
	 19,  # TEC Layer 1
	 19,  # TEC Layer 1
	 20,  # TEC Layer 2
	 20,  # TEC Layer 2
	 20,  # TEC Layer 2
	 21,  # TEC Layer 3
	 21,  # TEC Layer 3
	 21,  # TEC Layer 3
	 22,  # TEC Layer 4
	 22,  # TEC Layer 4
	 22,  # TEC Layer 4
	 23,  # TEC Layer 5
	 23,  # TEC Layer 5
	 23,  # TEC Layer 5
	 24,  # TEC Layer 6
	 24,  # TEC Layer 6
	 24,  # TEC Layer 6
	 25,  # TEC Layer 7
	 25,  # TEC Layer 7
	 26,  # TEC Layer 8
	 26,  # TEC Layer 8
	 27,  # TEC Layer 9
	 27,  # TEC Layer 9
	113,  # Barrel Wall
	114,  # Endcap Wall : 4.86<eta<4.91
	114,  # Endcap Wall : 4.82<eta<4.86
	114,  # Endcap Wall : 4.40<eta<4.82
	114,  # Endcap Wall : 4.00<eta<4.40
	114,  # Endcap Wall : 3.71<eta<4.00
	114,  # Endcap Wall : 3.61<eta<3.71
	114,  # Endcap Wall : 3.30<eta<3.61
	114,  # Endcap Wall : 3.05<eta<3.30
	114,  # Endcap Wall : 2.95<eta<3.05
	114,  # Endcap Wall : 2.75<eta<2.95
	114   # Endcap Wall
        ),

        # Lower limit on Radius or Z
        FudgeMin = cms.vdouble(
      	 0.0,  # Pixel Barrel services 
      	 4.2,  # Pixel Barrel services 
      	 5.1,  # Pixel Barrel services 
      	 7.1,  # Pixel Barrel services 
      	 8.2,  # Pixel Barrel services 
      	10.0,  # Pixel Barrel services 
      	11.0,  # Pixel Barrel services
	 0.0,  # Pixel Wall 
	29.0,  # Pixel Wall 
	 0.0,  # Pixel endcap services
	10.0,  # Pixel endcap services
	16.0,  # Pixel endcap services
	35.0,  # TIB1 services  
	35.0,  # TIB2 services  
	35.0,  # TIB3 services  
	35.0,  # TIB4 services  
	34.0,  # TID Layer 1
	34.0,  # TID Layer 2
	34.0,  # TID Layer 3
	47.5,  # TID outside services
	22.0,  # TID outside services
	27.5,  # TOB inside services
	72.0,  # TOB inside services
	 0.0,  # TOB Layer 1
	18.0,  # TOB Layer 1
	36.0,  # TOB Layer 1
	55.0,  # TOB Layer 1
	 0.0,  # TOB Layer 2
	18.0,  # TOB Layer 2
	36.0,  # TOB Layer 2
	55.0,  # TOB Layer 2
	 0.0,  # TOB Layer 3
	18.0,  # TOB Layer 3
	36.0,  # TOB Layer 3
	55.0,  # TOB Layer 3
	 0.0,  # TOB Layer 4
	18.0,  # TOB Layer 4
	36.0,  # TOB Layer 4
	55.0,  # TOB Layer 4
	 0.0,  # TOB Layer 5
	18.0,  # TOB Layer 5
	36.0,  # TOB Layer 5
	55.0,  # TOB Layer 5
	 0.0,  # TOB Layer 6
	18.0,  # TOB Layer 6
	36.0,  # TOB Layer 6
	55.0,  # TOB Layer 6
	55.0,  # TOB services
	62.0,  # TOB services
	78.0,  # TOB services
	 0.0,  # TEC Layer 1
	24.0,  # TEC Layer 1
	34.0,  # TEC Layer 1
	 0.0,  # TEC Layer 2
	24.0,  # TEC Layer 2
	34.0,  # TEC Layer 2
	 0.0,  # TEC Layer 3
	24.0,  # TEC Layer 3
	34.0,  # TEC Layer 3
	 0.0,  # TEC Layer 4
	32.0,  # TEC Layer 4
	41.0,  # TEC Layer 4
	 0.0,  # TEC Layer 5
	32.0,  # TEC Layer 5
	41.0,  # TEC Layer 5
	 0.0,  # TEC Layer 6
	32.0,  # TEC Layer 6
	41.0,  # TEC Layer 6
	 0.0,  # TEC Layer 7
	32.0,  # TEC Layer 7
	 0.0,  # TEC Layer 8
	32.0,  # TEC Layer 8
	 0.0,  # TEC Layer 9
	32.0,  # TEC Layer 9
       120.0,  # Barrel wall
         4.42, # Endcap Wall : 4.86<eta<4.91
	 4.65, # Endcap Wall : 4.82<eta<4.86
	 4.84, # Endcap Wall : 4.40<eta<4.82
	 7.37, # Endcap Wall : 4.00<eta<4.40
	10.99, # Endcap Wall : 3.71<eta<4.00
	14.70, # Endcap Wall : 3.61<eta<3.71
	16.24, # Endcap Wall : 3.30<eta<3.61
	22.00, # Endcap Wall : 3.05<eta<3.30
	28.50, # Endcap Wall : 2.95<eta<3.05
        31.50, # Endcap wall : 2.75<eta<2.95
        36.0   # Endcap wall
         ),

        # Upper limit on Radius or Z
        FudgeMax = cms.vdouble(
      	  4.2,  # Pixel Barrel services 
      	  5.1,  # Pixel Barrel services 
      	  7.1,  # Pixel Barrel services 
      	  8.2,  # Pixel Barrel services 
      	 10.0,  # Pixel Barrel services 
      	 11.0,  # Pixel Barrel services
       	 11.9,  # Pixel Barrel services 
	 29.0,  # Pixel Wall 
	 33.5,  # Pixel Wall 
	 10.0,  # Pixel endcap services
	 11.0,  # Pixel endcap services
	 18.0,  # Pixel endcap services
	 68.0,  # TIB1 services  
	 68.0,  # TIB2 services  
	 68.0,  # TIB3 services  
	 68.0,  # TIB4 services  
	 42.0,  # TID Layer 1
	 42.0,  # TID Layer 2
	 42.0,  # TID Layer 3
	 54.0,  # TID outside services
	 24.0,  # TID outside services
	 30.5,  # TOB inside services
        110.0,  # TOB inside services
	 18.0,  # TOB Layer 1
	 30.0,  # TOB Layer 1
	 46.0,  # TOB Layer 1
	108.0,  # TOB Layer 1
	 18.0,  # TOB Layer 2
	 30.0,  # TOB Layer 2
	 46.0,  # TOB Layer 2
	108.0,  # TOB Layer 2
	 18.0,  # TOB Layer 3
	 30.0,  # TOB Layer 3
	 46.0,  # TOB Layer 3
	108.0,  # TOB Layer 3
	 18.0,  # TOB Layer 4
	 30.0,  # TOB Layer 4
	 46.0,  # TOB Layer 4
	108.0,  # TOB Layer 4
	 18.0,  # TOB Layer 5
	 30.0,  # TOB Layer 5
	 46.0,  # TOB Layer 5
	108.0,  # TOB Layer 5
	 18.0,  # TOB Layer 6
	 30.0,  # TOB Layer 6
	 46.0,  # TOB Layer 6
	108.0,  # TOB Layer 6
	 60.0,  # TOB services
	 78.0,  # TOB services
	 92.0,  # TOB services
	 24.0,  # TEC Layer 1
	 34.0,  # TEC Layer 1
	 39.0,  # TEC Layer 1
	 24.0,  # TEC Layer 2
	 34.0,  # TEC Layer 2
	 39.0,  # TEC Layer 2
	 24.0,  # TEC Layer 3
	 34.0,  # TEC Layer 3
	 39.0,  # TEC Layer 3
	 32.0,  # TEC Layer 4
	 40.0,  # TEC Layer 4
	 46.0,  # TEC Layer 4
	 32.0,  # TEC Layer 5
	 40.0,  # TEC Layer 5
	 46.0,  # TEC Layer 5
	 32.0,  # TEC Layer 6
	 40.0,  # TEC Layer 6
	 46.0,  # TEC Layer 6
	 32.0,  # TEC Layer 7
	 60.0,  # TEC Layer 7
	 32.0,  # TEC Layer 8
	 60.0,  # TEC Layer 8
	 32.0,  # TEC Layer 9
	 60.0,  # TEC Layer 9
	301.0,  # Barrel wall
          4.65, # Endcap Wall : 4.86<eta<4.91
	  4.84, # Endcap Wall : 4.82<eta<4.86
	  7.37, # Endcap Wall : 4.40<eta<4.82
	 10.99, # Endcap Wall : 4.00<eta<4.40
	 14.70, # Endcap Wall : 3.71<eta<4.00
	 16.24, # Endcap Wall : 3.61<eta<3.71
	 22.00, # Endcap Wall : 3.30<eta<3.61
	 28.50, # Endcap Wall : 3.05<eta<3.30
	 31.50, # Endcap Wall : 2.95<eta<3.05
	 36.00, # Endcap Wall : 2.75<eta<2.95
	120.0   # Endcap wall
         ),

        # Fudge factor on x/X0
        FudgeFactor = cms.vdouble(
	0.00,  # Pixel Barrel services 
	2.50,  # Pixel Barrel services 
	0.00,  # Pixel Barrel services 
	2.70,  # Pixel Barrel services 
	0.00,  # Pixel Barrel services 
	2.80,  # Pixel Barrel services 
	0.50,  # Pixel Barrel services 
	0.27,  # Pixel Wall
	1.90,  # Pixel Wall 
	1.60,  # Pixel endcap services
	1.30,  # Pixel endcap services
	0.70,  # Pixel endcap services
	1.45,  # TIB1 services  
	1.45,  # TIB2 services  
	1.45,  # TIB3 services  
	1.45,  # TIB4 services  
	2.00,  # TID Layer 1
	2.00,  # TID Layer 2
	2.00,  # TID Layer 3
	2.50,  # TID outside services
	1.50,  # TID outside services
	4.00,  # TOB inside services
	5.50,  # TOB inside services
	0.70,  # TOB Layer 1
	2.00,  # TOB Layer 1
	2.00,  # TOB Layer 1
	2.00,  # TOB Layer 1
	0.70,  # TOB Layer 2
	2.00,  # TOB Layer 2
	2.00,  # TOB Layer 2
	2.00,  # TOB Layer 2
	0.70,  # TOB Layer 3
	2.00,  # TOB Layer 3
	2.00,  # TOB Layer 3
	2.00,  # TOB Layer 3
	0.70,  # TOB Layer 4
	2.00,  # TOB Layer 4
	2.00,  # TOB Layer 4
	2.00,  # TOB Layer 4
	0.70,  # TOB Layer 5
	2.00,  # TOB Layer 5
	2.00,  # TOB Layer 5
	2.00,  # TOB Layer 5
	0.70,  # TOB Layer 6
	2.00,  # TOB Layer 6
	2.00,  # TOB Layer 6
	2.00,  # TOB Layer 6
	0.50,  # TOB services
	1.50,  # TOB services
	1.80,  # TOB services
	 2.0,  # TEC Layer 1
	 0.8,  # TEC Layer 1
	 1.6,  # TEC Layer 1
	 2.0,  # TEC Layer 2
	 0.8,  # TEC Layer 2
	 1.6,  # TEC Layer 2
	 2.0,  # TEC Layer 3
	 0.8,  # TEC Layer 3
	 1.6,  # TEC Layer 3
	 2.3,  # TEC Layer 4
	 0.6,  # TEC Layer 4
	 1.4,  # TEC Layer 4
	 2.3,  # TEC Layer 5
	 0.6,  # TEC Layer 5
	 1.4,  # TEC Layer 5
	 2.5,  # TEC Layer 6
	 0.6,  # TEC Layer 6
	 1.4,  # TEC Layer 6
	 2.7,  # TEC Layer 7
	 0.6,  # TEC Layer 7
	 3.0,  # TEC Layer 8
	 0.6,  # TEC Layer 8
	 3.0,  # TEC Layer 9
	 0.6,  # TEC Layer 9
         3.8,  # Barrel wall
        18.74,  # Endcap Wall : 4.86<eta<4.91
	 2.30,  # Endcap Wall : 4.82<eta<4.86
	 0.604, # Endcap Wall : 4.40<eta<4.82
	 0.424, # Endcap Wall : 4.00<eta<4.40
	 0.327, # Endcap Wall : 3.71<eta<4.00
	 0.591, # Endcap Wall : 3.61<eta<3.71
	 7.00,  # Endcap Wall : 3.30<eta<3.61
	 4.40,  # Endcap Wall : 3.05<eta<3.30
	 3.30,  # Endcap Wall : 2.95<eta<3.05
	 1.40,  # Endcap Wall : 2.75<eta<2.95
         1.60   # Endcap wall
        )
        
    )

)

