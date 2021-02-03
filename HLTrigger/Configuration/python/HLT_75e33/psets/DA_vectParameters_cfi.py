import FWCore.ParameterSet.Config as cms

DA_vectParameters = cms.PSet(
    TkDAClusParameters = cms.PSet(
        Tmin = cms.double(2.0),
        Tpurge = cms.double(2.0),
        Tstop = cms.double(0.5),
        convergence_mode = cms.int32(0),
        coolingFactor = cms.double(0.6),
        d0CutOff = cms.double(3.0),
        delta_highT = cms.double(0.01),
        delta_lowT = cms.double(0.001),
        dzCutOff = cms.double(3.0),
        uniquetrkweight = cms.double(0.8),
        vertexSize = cms.double(0.006),
        zmerge = cms.double(0.01),
        zrange = cms.double(4.0)
    ),
    algorithm = cms.string('DA_vect')
)