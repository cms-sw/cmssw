import FWCore.ParameterSet.Config as cms

source = cms.Source("EmptySource")

generator = cms.EDFilter("ReggeGribovPartonMCGeneratorFilter",
                    beammomentum = cms.double(4000),
                    targetmomentum = cms.double(-4000),
                    beamid = cms.int32(1),
                    targetid = cms.int32(1),
                    model = cms.int32(0),
                    paramFileName = cms.string("Configuration/Generator/data/ReggeGribovPartonMC.param")
                    )


configurationMetadata = cms.untracked.PSet(
    version = cms.untracked.string('$Revision: 1.1 $'),
    name = cms.untracked.string('$Source: /local/reps/CMSSW/CMSSW/GeneratorInterface/ReggeGribovPartonMCInterface/python/ReggeGribovPartonMC_cfi.py,v $'),
    annotation = cms.untracked.string('ReggeGribovMC generator')
    )





