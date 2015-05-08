import FWCore.ParameterSet.Config as cms

#this will load the auto magnetic field producer reading the current from the DB
# and loading the best map available for that current as specified in the file 
from MagneticField.Engine.autoMagneticFieldProducer_cfi import *

# Parabolic parametrized magnetic field used for track building
import MagneticField.ParametrizedEngine.ParabolicParametrizedField_cfi
ParabolicParametrizedMagneticFieldProducer38T = MagneticField.ParametrizedEngine.ParabolicParametrizedField_cfi.ParametrizedMagneticFieldProducer.clone( label = "ParabolicMf38T" )

AutoParabolicParametrizedMagneticFieldProducer = cms.ESProducer("AutoMagneticFieldESProducer",
    # if positive, set B value (in kGauss), overriding the current reading from DB
    valueOverride = cms.int32(-1),
    nominalCurrents = cms.untracked.vint32(-1, 0,9558,14416,16819,18268,19262),
    mapLabels = cms.untracked.vstring("ParabolicMf38T",
                                      "slave_0", "slave_20", "slave_30", "slave_35", 
                                      "ParabolicMf38T", "slave_40" ),
   label = cms.untracked.string('ParabolicMf')
)

es_prefer_ParabolicMf = cms.ESPrefer("AutoMagneticFieldESProducer", "AutoParabolicParametrizedMagneticFieldProducer")
