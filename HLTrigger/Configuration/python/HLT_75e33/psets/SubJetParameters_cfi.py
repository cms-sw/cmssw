import FWCore.ParameterSet.Config as cms

SubJetParameters = cms.PSet(
    nFilt = cms.int32(2),
    rcut_factor = cms.double(0.5),
    zcut = cms.double(0.1)
)