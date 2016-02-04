import FWCore.ParameterSet.Config as cms

from Configuration.StandardSequences.MagneticField_cff import *

from TrackPropagation.SteppingHelixPropagator.SteppingHelixPropagatorAlong_cfi import *
from TrackPropagation.SteppingHelixPropagator.SteppingHelixPropagatorOpposite_cfi import *
from TrackPropagation.SteppingHelixPropagator.SteppingHelixPropagatorAny_cfi import *
from TrackPropagation.SteppingHelixPropagator.SteppingHelixPropagatorHLT_cff import *

localUniform = cms.ESProducer("UniformMagneticFieldESProducer",
    ZFieldInTesla = cms.double(0.0)
)

es_prefer_localUniform = cms.ESPrefer("UniformMagneticFieldESProducer","localUniform")
#   es_prefer magfield = XMLIdealGeometryESSource{}
SteppingHelixPropagatorAny.useInTeslaFromMagField = True
SteppingHelixPropagatorAny.SetVBFPointer = True

SteppingHelixPropagatorAlong.useInTeslaFromMagField = True
SteppingHelixPropagatorAlong.SetVBFPointer = True

SteppingHelixPropagatorOpposite.useInTeslaFromMagField = True
SteppingHelixPropagatorOpposite.SetVBFPointer = True

# HLT parts
SteppingHelixPropagatorL2Any.useInTeslaFromMagField = True
SteppingHelixPropagatorL2Any.SetVBFPointer = True

SteppingHelixPropagatorL2Along.useInTeslaFromMagField = True
SteppingHelixPropagatorL2Along.SetVBFPointer = True

SteppingHelixPropagatorL2Opposite.useInTeslaFromMagField = True
SteppingHelixPropagatorL2Opposite.SetVBFPointer = True


VolumeBasedMagneticFieldESProducer.label = 'VolumeBasedMagneticField'

