import FWCore.ParameterSet.Config as cms

from IOMC.EventVertexGenerators.VtxSmearedParameters_cfi import *

# default definition of common parameters

common_beam_direction_parameters = cms.PSet(
    BeamPosition = cms.double(0.),
    MinEta = cms.double(0.),
    MaxEta = cms.double(1.5),
    MinPhi = cms.double(-3.14159265358979323846),
    MaxPhi = cms.double(3.14159265358979323846)
)

#
# this module takes input in the units of *cm* and *radian*!!!
#

VtxSmeared = cms.EDProducer("BeamProfileVtxGenerator",
    common_beam_direction_parameters,
    VtxSmearedCommon,
    BeamMeanX       = cms.double(0.0),
    BeamMeanY       = cms.double(0.0),
    BeamSigmaX      = cms.double(0.0001),
    BeamSigmaY      = cms.double(0.0001),
    Psi             = cms.double(999.9),
    GaussianProfile = cms.bool(True),
    BinX       = cms.int32(50),
    BinY       = cms.int32(50),
    File       = cms.string('beam.profile'),
    UseFile    = cms.bool(False),
    TimeOffset = cms.double(0.)                      
)



