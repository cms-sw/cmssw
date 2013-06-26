import FWCore.ParameterSet.Config as cms

#
# use this cff to switch off the mag field 
#
UniformMagneticFieldESProducer = cms.ESProducer("UniformMagneticFieldESProducer",
    ZFieldInTesla = cms.double(0.0)
)

es_prefer_UniformMagneticFieldESProducer = cms.ESPrefer("UniformMagneticFieldESProducer")


