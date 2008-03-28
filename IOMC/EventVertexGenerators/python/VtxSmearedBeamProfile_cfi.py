import FWCore.ParameterSet.Config as cms

#
# this module takes inpuut in the units of *cm* and *radian*!!!
#
VtxSmeared = cms.EDFilter("BeamProfileVtxGenerator",
    common_beam_direction_parameters,
    BeamSigmaX = cms.untracked.double(0.0001),
    BeamSigmaY = cms.untracked.double(0.0001),
    BeamMeanY = cms.untracked.double(0.0),
    BeamMeanX = cms.untracked.double(0.0),
    GaussianProfile = cms.untracked.bool(True),
    BinY = cms.untracked.int32(50),
    BinX = cms.untracked.int32(50),
    File = cms.untracked.string('beam.profile'),
    UseFile = cms.untracked.bool(False)
)


