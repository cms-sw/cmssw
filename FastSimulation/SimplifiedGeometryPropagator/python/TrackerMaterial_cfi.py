import FWCore.ParameterSet.Config as cms
from Configuration.Eras.Modifier_phase1Pixel_cff import phase1Pixel #to configure phase1 geoemtry

# Do not change the order of the interaction models unless you know what you are doing.
# Not used at the moment: "muonBremsstrahlung", "nuclearInteractionFTF"
#_trackerMaterialInteractionModels = cms.untracked.vstring("trackerSimHits")
_trackerMaterialInteractionModels = cms.untracked.vstring("pairProduction", "nuclearInteraction", "bremsstrahlung", "energyLoss", "multipleScattering", "trackerSimHits")

# Material effects to be simulated in the tracker material and associated cuts 
TrackerMaterialBlock = cms.PSet(
    TrackerMaterial = cms.PSet(
        #magneticFieldZ = cms.untracked.double(0.),
        maxRadius = cms.untracked.double(150.),
        maxZ = cms.untracked.double(325.),
        useTrackerRecoGeometryRecord = cms.untracked.bool(True),
        trackerAlignmentLabel = cms.untracked.string("MisAligned"),

        #############
        ### Hack to interface "old" calorimetry with "new" propagation in tracker
        #############
        # Outer boundaries of tracker -> Beginning of calorimetry
        # This includes: Preshower, ECAL barrel/forward entrance
        # Radius/z must be slightly smaller than actual geometry (e.g. 0.1cm)       
        trackerBarrelBoundary = cms.PSet(
            radius = cms.untracked.double(128.9),
            limits = cms.untracked.vdouble(0.0, 303.353),
            thickness = cms.untracked.vdouble(1.),
            interactionModels = cms.untracked.vstring()
        ),
        trackerForwardBoundary = cms.PSet(
            z = cms.untracked.double(303.253),
            limits = cms.untracked.vdouble(0.0, 129.),
            thickness = cms.untracked.vdouble(1.),
            interactionModels = cms.untracked.vstring()
        ),
        #############
        ### End Hack
        #############
        
        # The tracker layers
        BarrelLayers = cms.VPSet(
            ########### Beam Pipe ###########
            #PIPE
            cms.PSet(
                radius = cms.untracked.double(3.003),
                limits = cms.untracked.vdouble(0.0, 28.3),
                thickness = cms.untracked.vdouble(0.0024),
                interactionModels = _trackerMaterialInteractionModels
            ),
            ########### The Pixel Barrel layers 1-3 ###########
            #PIXB1
            cms.PSet(
                #radius = cms.untracked.double(4.425),
                limits = cms.untracked.vdouble(0.0, 28.391),
                thickness = cms.untracked.vdouble(0.0217),
                activeLayer = cms.untracked.string("BPix1"),
                interactionModels = _trackerMaterialInteractionModels
            ),
            #PIXB2
            cms.PSet(
                #radius = cms.untracked.double(7.312),
                limits = cms.untracked.vdouble(0.0, 28.391),
                thickness = cms.untracked.vdouble(0.0217),
                activeLayer = cms.untracked.string("BPix2"),
                interactionModels = _trackerMaterialInteractionModels
            ),
            #PIXB3
            cms.PSet(
                #radius = cms.untracked.double(10.177),
                limits = cms.untracked.vdouble(0.0, 28.391),
                thickness = cms.untracked.vdouble(0.0217),
                activeLayer = cms.untracked.string("BPix3"),
                interactionModels = _trackerMaterialInteractionModels
            ),

            ########### Pixel Outside walls and cables (barrel) ###########
            #PIXBOut5
            cms.PSet(
                radius = cms.untracked.double(17.6),
                limits = cms.untracked.vdouble(0.0, 27.5, 32.0, 65.0),
                thickness = cms.untracked.vdouble(0.0135, 0.095, 0.050),
                interactionModels = _trackerMaterialInteractionModels
            ),
            ########### Tracker Inner barrel layers 1-4 ###########
            #TIB1
            cms.PSet(
                #radius = cms.untracked.double(25.767),
                limits = cms.untracked.vdouble(0.0, 35.0, 65.254),
                thickness = cms.untracked.vdouble(0.053, 0.0769),
                activeLayer = cms.untracked.string("TIB1"),
                interactionModels = _trackerMaterialInteractionModels
            ),
            #TIB2
            cms.PSet(
                #radius = cms.untracked.double(34.104),
                limits = cms.untracked.vdouble(0.0, 35.0, 65.231),
                thickness = cms.untracked.vdouble(0.053, 0.0769),
                activeLayer = cms.untracked.string("TIB2"),
                interactionModels = _trackerMaterialInteractionModels
            ),
            #TIB3
            cms.PSet(
                #radius = cms.untracked.double(41.974),
                limits = cms.untracked.vdouble(0.0, 35.0, 66.232),
                thickness = cms.untracked.vdouble(0.035, 0.0508),
                activeLayer = cms.untracked.string("TIB3"),
                interactionModels = _trackerMaterialInteractionModels
            ),
            #TIB4
            cms.PSet(
                #radius = cms.untracked.double(49.907),
                limits = cms.untracked.vdouble(0.0, 35.0, 66.355),
                thickness = cms.untracked.vdouble(0.04, 0.058),
                activeLayer = cms.untracked.string("TIB4"),
                interactionModels = _trackerMaterialInteractionModels
            ),
            ########### TOB inside wall (barrel) ###########
            #TOBCIn
            cms.PSet(
                radius = cms.untracked.double(55.1),
                limits = cms.untracked.vdouble(0.0, 27.5, 30.5, 72.0, 108.2),
                thickness = cms.untracked.vdouble(0.009, 0.036, 0.009, 0.0495),
                interactionModels = _trackerMaterialInteractionModels
            ),
            ########### # Tracker Outer barrel layers 1-6 ###########
            #TOB1
            cms.PSet(
                #radius = cms.untracked.double(60.937),
                limits = cms.untracked.vdouble(0.0, 18.0, 30.0, 36.0, 46.0, 55.0, 108.737),
                thickness = cms.untracked.vdouble(0.021, 0.06, 0.03, 0.06, 0.03, 0.06),
                activeLayer = cms.untracked.string("TOB1"),
                interactionModels = _trackerMaterialInteractionModels
            ),
            #TOB2
            cms.PSet(
                #radius = cms.untracked.double(69.322),
                limits = cms.untracked.vdouble(0.0, 18.0, 30.0, 36.0, 46.0, 55.0, 108.737),
                thickness = cms.untracked.vdouble(0.021, 0.06, 0.03, 0.06, 0.03, 0.06),
                activeLayer = cms.untracked.string("TOB2"),
                interactionModels = _trackerMaterialInteractionModels
            ),
            #TOB3
            cms.PSet(
                #radius = cms.untracked.double(78.081),
                limits = cms.untracked.vdouble(0.0, 18.0, 30.0, 36.0, 46.0, 55.0, 108.737),
                thickness = cms.untracked.vdouble(0.0154, 0.044, 0.022, 0.044, 0.022, 0.044),
                activeLayer = cms.untracked.string("TOB3"),
                interactionModels = _trackerMaterialInteractionModels
            ),
            #TOB4
            cms.PSet(
                #radius = cms.untracked.double(86.876),
                limits = cms.untracked.vdouble(0.0, 18.0, 30.0, 36.0, 46.0, 55.0, 108.737),
                thickness = cms.untracked.vdouble(0.0154, 0.044, 0.022, 0.044, 0.022, 0.044),
                activeLayer = cms.untracked.string("TOB4"),
                interactionModels = _trackerMaterialInteractionModels
            ),
            #TOB5
            cms.PSet(
                #radius = cms.untracked.double(96.569),
                limits = cms.untracked.vdouble(0.0, 18.0, 30.0, 36.0, 46.0, 55.0, 108.737),
                thickness = cms.untracked.vdouble(0.0154, 0.044, 0.022, 0.044, 0.022, 0.044),
                activeLayer = cms.untracked.string("TOB5"),
                interactionModels = _trackerMaterialInteractionModels
            ),
            #TOB6
            cms.PSet(
                #radius = cms.untracked.double(108.063),
                limits = cms.untracked.vdouble(0.0, 18.0, 30.0, 36.0, 46.0, 55.0, 108.737),
                thickness = cms.untracked.vdouble(0.0154, 0.044, 0.022, 0.044, 0.022, 0.044),
                activeLayer = cms.untracked.string("TOB6"),
                interactionModels = _trackerMaterialInteractionModels
            ),
           
            ########### Tracker Outer Barrel Outside Cables and walls (barrel) ###########
            #TBOut
            cms.PSet(
                radius = cms.untracked.double(120.0),
                limits = cms.untracked.vdouble(0.0, 120.0, 299.9),
                thickness = cms.untracked.vdouble(0.042, 0.1596),
                interactionModels = _trackerMaterialInteractionModels
            ),
        ),

        EndcapLayers = cms.VPSet(
            ########### Pixel Barrel Outside walls and cables (endcap) ###########
            #PIXBOut4
            cms.PSet(
                z = cms.untracked.double(28.7),
                limits = cms.untracked.vdouble(4.2, 5.1, 7.1, 8.2, 10.0, 11.0, 11.9, 16.5),
                thickness = cms.untracked.vdouble(0.100, 0.00, 0.108, 0.00, 0.112, 0.02, 0.04),
                interactionModels = _trackerMaterialInteractionModels
            ),
            #PIXBOut
            cms.PSet(
                z = cms.untracked.double(28.8),
                limits = cms.untracked.vdouble(3.8, 16.5),
                thickness = cms.untracked.vdouble(0.012),
                interactionModels = _trackerMaterialInteractionModels
            ),
            ########### Pixel Disks 1-2 ###########
            #PIXD1
            cms.PSet(
                limits = cms.untracked.vdouble(4.825, 16.598),
                thickness = cms.untracked.vdouble(0.058),
                activeLayer = cms.untracked.string("FPix1"),
                interactionModels = _trackerMaterialInteractionModels
            ),
            #PIXD2
            cms.PSet(
                limits = cms.untracked.vdouble(4.823, 16.598),
                thickness = cms.untracked.vdouble(0.058),
                activeLayer = cms.untracked.string("FPix2"),
                interactionModels = _trackerMaterialInteractionModels
            ),

            ########### Pixel Endcap outside cables ###########
            #PIXBOut6
            cms.PSet(
                z = cms.untracked.double(65.1),
                limits = cms.untracked.vdouble(6.5, 10.0, 11.0, 16.0, 17.61),
                thickness = cms.untracked.vdouble(0.150, 0.325, 0.250, 0.175),
                interactionModels = _trackerMaterialInteractionModels
            ),
            ########### Tracker Inner Barrel Outside Cables and walls (endcap) ###########
            #TIBEOut
            cms.PSet(
                z = cms.untracked.double(74.0),
                limits = cms.untracked.vdouble(22.5, 53.9),
                thickness = cms.untracked.vdouble(0.130),
                interactionModels = _trackerMaterialInteractionModels
            ),
            ########### Tracker Inner disks layers 1-3 ###########
            #TID1
            cms.PSet(
                limits = cms.untracked.vdouble(22.2, 34.0, 42.0, 53.940),
                thickness = cms.untracked.vdouble(0.04, 0.08, 0.04),
                activeLayer = cms.untracked.string("TID1"),
                interactionModels = _trackerMaterialInteractionModels
            ),
            #TID2
            cms.PSet(
                limits = cms.untracked.vdouble(22.2, 34.0, 42.0, 53.942), 
                thickness = cms.untracked.vdouble(0.04, 0.08, 0.04),
                activeLayer = cms.untracked.string("TID2"),
                interactionModels = _trackerMaterialInteractionModels
            ),
            #TID3
            cms.PSet(
                limits = cms.untracked.vdouble(22.2, 34.0, 42.0, 53.942), 
                thickness = cms.untracked.vdouble(0.055, 0.110, 0.055),
                activeLayer = cms.untracked.string("TID3"),
                interactionModels = _trackerMaterialInteractionModels
            ),
            ########### Tracker Inner Disks Outside Cables and walls (endcap) ###########
            #TIDEOut
            cms.PSet(
                z = cms.untracked.double(108.0),
                limits = cms.untracked.vdouble(22.0, 24.0, 47.5, 54.943),
                thickness = cms.untracked.vdouble(0.111, 0.074, 0.185),
                interactionModels = _trackerMaterialInteractionModels
            ),
            ########### Tracker Outer Barrel Outside Cables and walls (barrel and endcap) ###########
            #TOBEOut
            cms.PSet(
                z = cms.untracked.double(115.0),
                limits = cms.untracked.vdouble(55.0, 60.0, 62.0, 78.0, 92.0, 111.0),
                thickness = cms.untracked.vdouble(0.005, 0.009, 0.014, 0.016, 0.009),
                interactionModels = _trackerMaterialInteractionModels
            ),
            ########### Tracker EndCap disks layers 1-9 ###########
            #TEC1
            cms.PSet(
                limits = cms.untracked.vdouble(21.87, 24.0, 34.0, 39.0, 111.395),
                thickness = cms.untracked.vdouble(0.100, 0.040, 0.080, 0.050),
                activeLayer = cms.untracked.string("TEC1"),
                nuclearInteractionThicknessFactor = cms.untracked.double(1.2),
                interactionModels = _trackerMaterialInteractionModels
            ),
            #TEC2
            cms.PSet(
                limits = cms.untracked.vdouble(21.87, 24.0, 34.0, 39.0, 111.395),
                thickness = cms.untracked.vdouble(0.100, 0.040, 0.080, 0.050),
                activeLayer = cms.untracked.string("TEC2"),
                nuclearInteractionThicknessFactor = cms.untracked.double(1.2),
                interactionModels = _trackerMaterialInteractionModels
            ),
            #TEC3
            cms.PSet(
                limits = cms.untracked.vdouble(21.87, 24.0, 34.0, 39.0, 111.395),
                thickness = cms.untracked.vdouble(0.100, 0.040, 0.080, 0.050),
                activeLayer = cms.untracked.string("TEC3"),
                nuclearInteractionThicknessFactor = cms.untracked.double(1.2),
                interactionModels = _trackerMaterialInteractionModels
            ),
            #TEC4
            cms.PSet(
                limits = cms.untracked.vdouble(29.62, 32.0, 40.0, 41.0, 46.0, 111.395),
                thickness = cms.untracked.vdouble(0.115, 0.030, 0.050, 0.070, 0.050),
                activeLayer = cms.untracked.string("TEC4"),
                nuclearInteractionThicknessFactor = cms.untracked.double(1.2),
                interactionModels = _trackerMaterialInteractionModels
            ),
            #TEC5
            cms.PSet(
                limits = cms.untracked.vdouble(29.62, 32.0, 40.0, 41.0, 46.0, 111.395),
                thickness = cms.untracked.vdouble(0.115, 0.030, 0.050, 0.070, 0.050),
                activeLayer = cms.untracked.string("TEC5"),
                nuclearInteractionThicknessFactor = cms.untracked.double(1.2),
                interactionModels = _trackerMaterialInteractionModels
            ),
            #TEC6
            cms.PSet(
                limits = cms.untracked.vdouble(29.62, 32.0, 40.0, 41.0, 46.0, 111.395),
                thickness = cms.untracked.vdouble(0.125, 0.030, 0.050, 0.070, 0.050),
                activeLayer = cms.untracked.string("TEC6"),
                nuclearInteractionThicknessFactor = cms.untracked.double(1.2),
                interactionModels = _trackerMaterialInteractionModels
            ),
            #TEC7
            cms.PSet(
                limits = cms.untracked.vdouble(29.71, 32.0, 60.0, 111.395),
                thickness = cms.untracked.vdouble(0.135, 0.030, 0.050),
                activeLayer = cms.untracked.string("TEC7"),
                nuclearInteractionThicknessFactor = cms.untracked.double(1.2),
                interactionModels = _trackerMaterialInteractionModels
                ),
            #TEC8
            cms.PSet(
                limits = cms.untracked.vdouble(29.71, 32.0, 60.0, 111.395),
                thickness = cms.untracked.vdouble(0.150, 0.030, 0.050),
                activeLayer = cms.untracked.string("TEC8"),
                nuclearInteractionThicknessFactor = cms.untracked.double(1.2),
                interactionModels = _trackerMaterialInteractionModels
                ),
            #TEC9
            cms.PSet(
                limits = cms.untracked.vdouble(29.91, 32.0, 60.0, 111.395),
                thickness = cms.untracked.vdouble(0.150, 0.030, 0.050),
                activeLayer = cms.untracked.string("TEC9"),
                nuclearInteractionThicknessFactor = cms.untracked.double(1.2),
                interactionModels = _trackerMaterialInteractionModels
                ),
            ########### Tracker Endcaps Outside Cables and walls (endcaps) ###########
            #TEOut
            cms.PSet(
                z = cms.untracked.double(300.0),
                limits = cms.untracked.vdouble(4.42, 4.65, 4.84, 7.37, 10.99, 14.70, 16.24, 22.00, 28.50, 31.50, 36.0, 120.0),
                thickness = cms.untracked.vdouble(3.935, 0.483, 0.127, 0.089, 0.069, 0.124, 1.47, 0.924, 0.693, 0.294, 0.336),
                interactionModels = _trackerMaterialInteractionModels
                ),
            ),
        )
    )
#new phase1 geometry
phase1Pixel.toModify(TrackerMaterialBlock, TrackerMaterial = dict(
        BarrelLayers = TrackerMaterialBlock.TrackerMaterial.BarrelLayers[:4] + [
            #PIXB4                                                                                                                        
            cms.PSet(
                #radius = cms.untracked.double(16),                                                                                        
                limits = cms.untracked.vdouble(0.0, 28.391),
                thickness = cms.untracked.vdouble(0.0217),
                activeLayer = cms.untracked.string("BPix4"),
                interactionModels = _trackerMaterialInteractionModels
                ),] + TrackerMaterialBlock.TrackerMaterial.BarrelLayers[4:],
        EndcapLayers = TrackerMaterialBlock.TrackerMaterial.EndcapLayers[:4] + [
            #PIXD3                                                                                                                 
            cms.PSet(
                limits = cms.untracked.vdouble(4.823, 16.598),
                thickness = cms.untracked.vdouble(0.058),
                activeLayer = cms.untracked.string("FPix3"),
                interactionModels = _trackerMaterialInteractionModels
                ),] + TrackerMaterialBlock.TrackerMaterial.EndcapLayers[4:]
        ))
