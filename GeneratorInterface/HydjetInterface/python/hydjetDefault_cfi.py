import FWCore.ParameterSet.Config as cms

source = cms.Source("EmptySource")

from GeneratorInterface.HydjetInterface.hydjetDefaultParameters_cff import *
generator = cms.EDFilter("HydjetGeneratorFilter",
                         hydjetDefaultParameters,

                         maxEventsToPrint = cms.untracked.int32(0),
                         pythiaPylistVerbosity = cms.untracked.int32(0),
                         firstEvent = cms.untracked.uint32(1),
                         firstRun = cms.untracked.uint32(1),
                         
                         comEnergy = cms.double(4000.0),
                         
                         cFlag = cms.int32(0),
                         bFixed = cms.double(0),
                         bMin = cms.double(0),
                         bMax = cms.double(0)
                         )

configurationMetadata = cms.untracked.PSet(
    version = cms.untracked.string('$Revision: 1.3 $'),
    name = cms.untracked.string('$Source: /local/reps/CMSSW/CMSSW/GeneratorInterface/HydjetInterface/python/hydjetDefault_cfi.py,v $'),
    annotation = cms.untracked.string('Hydjet-B0 at 4TeV')
    )





