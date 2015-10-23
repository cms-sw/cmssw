import FWCore.ParameterSet.Config as cms
 
from IOMC.EventVertexGenerators.VtxSmearedParameters_cfi import *
VtxSmeared = cms.EDProducer("MixBoostEvtVtxGenerator",
                            useCF = cms.untracked.bool(True),
                            signalLabel = cms.InputTag("generator","unsmeared"),
                            mixLabel = cms.InputTag("mix","generatorunsmeared"),
                            Beta=cms.double(-0.434)
                            )
