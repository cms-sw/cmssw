import FWCore.ParameterSet.Config as cms

from GeneratorInterface.CepGenInterface.cepgenDefaultParameters_cff import *

generator = cms.EDFilter("CepGenGeneratorFilter",
    process = cms.PSet(
        name = cms.string('pptoff'),
        kinematics = cms.PSet(
            mode = cms.uint32(2),  # 1 = elastic, 2-3 = SD, 4 = DD
            structureFunctions = cms.PSet(
                name = cms.int32(301)  # see https://cepgen.hepforge.org/raw-modules
            ),
            cmEnergy = cms.double(13000.0),
            qt = cms.vdouble(0., 50.),
            rapidity = cms.vdouble(-6., 6.),
            eta = cms.vdouble(-2.5, 2.5),
            pt = cms.vdouble(25., 9999.),
            ptdiff = cms.vdouble(0., 500.)
        ),
        processParameters = cms.PSet(
            ktFactorised = cms.bool(True),
            pair = cms.uint32(13)
        )
    ),
    outputModules = cepgenOutputModules,
    maxEventsToPrint = cms.untracked.int32(0),
    verbosity = cms.untracked.int32(0)
)
