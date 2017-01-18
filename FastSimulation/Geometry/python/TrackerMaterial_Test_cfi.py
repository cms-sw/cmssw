import FWCore.ParameterSet.Config as cms

#_trackerMaterialInteractionModels = cms.untracked.vstring("simpleLayerHits","bremsstrahlung")
_trackerMaterialInteractionModels = cms.untracked.vstring("trackerSimHits","bremsstrahlung")

# Material effects to be simulated in the tracker material and associated cuts 
TrackerMaterialBlock = cms.PSet(
    TrackerMaterial = cms.PSet(
        magneticFieldZ = cms.untracked.double(3.8),
        useTrackerRecoGeometryRecord = cms.untracked.bool(False),
        trackerAlignmentLabel = cms.untracked.string("MisAligned"),
        interactionModels = cms.PSet(
            #simpleLayerHits = cms.PSet(
            #    className = cms.string("simpleLayerHits")
            #    ),
            trackerSimHits = cms.PSet(
                className = cms.string("trackerSimHits")
                ),
            bremsstrahlung = cms.PSet(
                className = cms.string("bremsstrahlung"),
                minPhotonEnergy = cms.double(0.1),
                minPhotonEnergyFraction = cms.double(0.005)
                )
            ),
        BarrelLayers = cms.VPSet(
            cms.PSet(
                radius = cms.untracked.double(10.),
                limits = cms.untracked.vdouble(0.0,1.),
                thickness = cms.untracked.vdouble(0.01),
                interactionModels = _trackerMaterialInteractionModels
                ),
            cms.PSet(
                radius = cms.untracked.double(50.),
                limits = cms.untracked.vdouble(0.0,5.),
                thickness = cms.untracked.vdouble(0.05),
                interactionModels = _trackerMaterialInteractionModels
                ),
            cms.PSet(
                radius = cms.untracked.double(100.),
                limits = cms.untracked.vdouble(00.0,100.),
                thickness = cms.untracked.vdouble(0.1),
                interactionModels = _trackerMaterialInteractionModels
                ),
            ),
        ForwardLayers = cms.VPSet(
            cms.PSet(
                z = cms.untracked.double(10.),
                limits = cms.untracked.vdouble(0.0,1.),
                thickness = cms.untracked.vdouble(0.01),
                interactionModels = _trackerMaterialInteractionModels
                ),
            cms.PSet(
                z = cms.untracked.double(50.),
                limits = cms.untracked.vdouble(0.0,5.),
                thickness = cms.untracked.vdouble(0.05),
                interactionModels = _trackerMaterialInteractionModels
                ),
            cms.PSet(
                z = cms.untracked.double(100.),
                limits = cms.untracked.vdouble(0.0,100.),
                thickness = cms.untracked.vdouble(0.1),
                interactionModels = _trackerMaterialInteractionModels
                ),
            ),
        )
    )
