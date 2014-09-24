import FWCore.ParameterSet.Config as cms


# This cfi contains everything needed to use the VolumeBased magnetic
# field engine version 090322 (based on 2007 geometry, model with extended R and Z)
# with separate tables for different sectors


ParametrizedMagneticFieldProducer = cms.ESProducer("ParametrizedMagneticFieldProducer",
    version = cms.string('OAE_1103l_071212'),
    parameters = cms.PSet(
        BValue = cms.string('3_8T')
    ),
    label = cms.untracked.string('parametrizedField')
)

#Apply scaling factors
from MagneticField.Engine.ScalingFactors_090322_2pi_090520_cfi import *

VolumeBasedMagneticFieldESProducer = cms.ESProducer("VolumeBasedMagneticFieldESProducerFromDB",
    label = cms.untracked.string(''),
    debugBuilder = cms.untracked.bool(False),
    valueOverride = cms.int32(-1),
                                                    
    # Parameters to be moved to the config DB                                                
    version = cms.string('grid_1103l_090322_3_8t'),
    geometryVersion = cms.int32(90322),
    fieldScaling,
    paramLabel = cms.string('parametrizedField'), #FIXME
    paramData = cms.vdouble(3.8),#FIXME
    cacheLastVolume = cms.untracked.bool(True),#FIXME
                                                    
    gridFiles = cms.VPSet(
        cms.PSet( # Default tables, replicate sector 1
            volumes   = cms.string('1-312'),
            sectors   = cms.string('0') ,
            master    = cms.int32(1),
            path      = cms.string('grid.[v].bin'),
        ),

        cms.PSet( # Specific volumes in Barrel, sector 3
            volumes   = cms.string('176-186,231-241,286-296'),
            sectors   = cms.string('3') ,
            master    = cms.int32(3),
            path      = cms.string('S3/grid.[v].bin'),
        ),

        cms.PSet( # Specific volumes in Barrel, sector 4
            volumes   = cms.string('176-186,231-241,286-296'),
            sectors   = cms.string('4') ,
            master    = cms.int32(4),
            path      = cms.string('S4/grid.[v].bin'),
        ),

        cms.PSet(  # Specific volumes in Barrel and endcaps, sector 9
            volumes   = cms.string('14,15,20,21,24-27,32,33,40,41,48,49,56,57,62,63,70,71,286-296'),
            sectors   = cms.string('9') ,
            master    = cms.int32(9),
            path      = cms.string('S9/grid.[v].bin'),
        ),

        cms.PSet(  # Specific volumes in Barrel and endcaps, sector 10
            volumes   = cms.string('14,15,20,21,24-27,32,33,40,41,48,49,56,57,62,63,70,71,286-296'),
            sectors   = cms.string('10') ,
            master    = cms.int32(10),
            path      = cms.string('S10/grid.[v].bin'),
        ),
                                                        
        cms.PSet( # Specific volumes in Barrel and endcaps, sector 11
            volumes   = cms.string('14,15,20,21,24-27,32,33,40,41,48,49,56,57,62,63,70,71,286-296'),
            sectors   = cms.string('11') ,
            master    = cms.int32(11),
            path      = cms.string('S11/grid.[v].bin'),
        ),
    )
)

