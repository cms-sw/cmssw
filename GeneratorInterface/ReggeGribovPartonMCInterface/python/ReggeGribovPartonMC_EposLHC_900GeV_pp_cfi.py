import FWCore.ParameterSet.Config as cms

from GeneratorInterface.ReggeGribovPartonMCInterface.ReggeGribovPartonMC_AdvancedParameters_cfi import *

generator = cms.EDFilter("ReggeGribovPartonMCGeneratorFilter",
                    ReggeGribovPartonMCAdvancedParameters,
                    beammomentum = cms.double(450),
                    targetmomentum = cms.double(-450),
                    beamid = cms.int32(1),
                    targetid = cms.int32(1),
                    model = cms.int32(0)
                    )


configurationMetadata = cms.untracked.PSet(
    version = cms.untracked.string('$Revision: 1.2 $'),
    name = cms.untracked.string('$Source: /local/reps/CMSSW/CMSSW/GeneratorInterface/ReggeGribovPartonMCInterface/python/ReggeGribovPartonMC_EposLHC_900GeV_pp_cfi.py,v $'),
    annotation = cms.untracked.string('ReggeGribovMC generator')
    )





