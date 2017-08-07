import FWCore.ParameterSet.Config as cms

from FastSimulation.SimplifiedGeometryPropagator.TrackerMaterial_cfi import TrackerMaterialBlock 

#############
### Hack to interface "old" calorimetry with "new" propagation in tracker
#############

CaloMaterialBlock = cms.PSet(
    CaloMaterial = cms.PSet(
        maxRadius = cms.untracked.double(500.),
        maxZ = cms.untracked.double(1200.),
        interactionModels = cms.untracked.vstring(),
        
        ######
        # The calorimetry
        # Positions used from old ParticlePropagator. Do not really agree with the CMS ECAL/HCAL TDR values...
        ######

        # Coverage usually provided as eta, e.g. barrel ECAL abs(eta) < 1.479
        # Use definition of pseurorapidity: theta = 2*arctan(e^-eta)
        # And theta = tan(R/z)
        # Solve for z to get range of barrel ECAL (z < 306.227)
        BarrelLayers = cms.VPSet(
            ########### ECAL ###########
            cms.PSet(
                radius = cms.untracked.double(129.0),
                limits = cms.untracked.vdouble(0.0, 306.227),
                thickness = cms.untracked.vdouble(1.),
                interactionModels = cms.untracked.vstring(),
                caloType = cms.untracked.string("ECAL")
            ),
            ########### HCAL ###########
            cms.PSet(
                radius = cms.untracked.double(177.5),
                limits = cms.untracked.vdouble(0.0, 335.0),
                thickness = cms.untracked.vdouble(1.),
                interactionModels = cms.untracked.vstring(),
                caloType = cms.untracked.string("HCAL")
            ),
        ),

		ForwardLayers = cms.VPSet(
            ########### PreShowerLayer1 ###########
            cms.PSet(
                z = cms.untracked.double(303.353),
                limits = cms.untracked.vdouble(45., 125.),
                thickness = cms.untracked.vdouble(1.),
                interactionModels = cms.untracked.vstring(),
                caloType = cms.untracked.string("PRESHOWER1")
            ),
            ########### PreShowerLayer2 ###########
            cms.PSet(
                z = cms.untracked.double(307.838),
                limits = cms.untracked.vdouble(45., 125.),
                thickness = cms.untracked.vdouble(1.),
                interactionModels = cms.untracked.vstring(),
                caloType = cms.untracked.string("PRESHOWER2")
            ),
            ########### ECAL ###########
            cms.PSet(
                z = cms.untracked.double(320.9),
                limits = cms.untracked.vdouble(31.822, 152.9),
                thickness = cms.untracked.vdouble(1.),
                interactionModels = cms.untracked.vstring(),
                caloType = cms.untracked.string("ECAL")
            ),
            ########### HCAL ###########
            cms.PSet(
                z = cms.untracked.double(400.458),
                limits = cms.untracked.vdouble(39.712, 300.),
                thickness = cms.untracked.vdouble(1.),
                interactionModels = cms.untracked.vstring(),
                caloType = cms.untracked.string("HCAL")
            ),
            ########### VFCAL ###########
            cms.PSet(
                z = cms.untracked.double(1110.0),
                limits = cms.untracked.vdouble(14.957, 110.074),
                thickness = cms.untracked.vdouble(1.),
                interactionModels = cms.untracked.vstring(),
                caloType = cms.untracked.string("VFCAL")
            ),
        )
    )
)
    
if hasattr(TrackerMaterialBlock.TrackerMaterial, 'magneticFieldZ'):
    CaloMaterialBlock.CaloMaterial.magneticFieldZ = TrackerMaterialBlock.TrackerMaterial.magneticFieldZ
    
