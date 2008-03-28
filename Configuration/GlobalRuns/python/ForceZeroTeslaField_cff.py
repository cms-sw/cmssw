import FWCore.ParameterSet.Config as cms

localUniform = cms.ESProducer("UniformMagneticFieldESProducer",
    ZFieldInTesla = cms.double(0.0)
)

es_prefer_localUniform = cms.ESPrefer("UniformMagneticFieldESProducer","localUniform")
#   es_prefer magfield = XMLIdealGeometryESSource{}
SteppingHelixPropagatorAny.useInTeslaFromMagField = True
SteppingHelixPropagatorAlong.useInTeslaFromMagField = True
SteppingHelixPropagatorOpposite.useInTeslaFromMagField = True
SteppingHelixPropagatorAny.SetVBFPointer = True
SteppingHelixPropagatorAlong.SetVBFPointer = True
SteppingHelixPropagatorOpposite.SetVBFPointer = True
VolumeBasedMagneticFieldESProducer.label = 'VolumeBasedMagneticField'

