import FWCore.ParameterSet.Config as cms

from IOMC.EventVertexGenerators.VtxSmearedParameters_cfi import *

# default definition of common parameters

common_beam_direction_parameters = cms.PSet(
    BeamPosition = cms.untracked.double(0.),
    MinEta = cms.untracked.double(0.),
    MaxEta = cms.untracked.double(1.5),
    MinPhi = cms.untracked.double(-3.14159265358979323846),
    MaxPhi = cms.untracked.double(3.14159265358979323846)
)

#
# this module takes input in the units of *cm* and *radian*!!!
#

VtxSmeared = cms.EDFilter("BeamProfileVtxGenerator",
    common_beam_direction_parameters,
    VtxSmearedCommon,
    BeamMeanX = cms.untracked.double(0.0),
    BeamMeanY = cms.untracked.double(0.0),
    BeamSigmaX = cms.untracked.double(0.0001),
    BeamSigmaY = cms.untracked.double(0.0001),
    GaussianProfile = cms.untracked.bool(True),
    BinX = cms.untracked.int32(50),
    BinY = cms.untracked.int32(50),
    File = cms.untracked.string('beam.profile'),
    UseFile = cms.untracked.bool(False),
    TimeOffset = cms.double(0.)                      
)



