import FWCore.ParameterSet.Config as cms
 
from IOMC.EventVertexGenerators.VtxSmearedParameters_cfi import *
VtxSmeared = cms.EDProducer("MixEvtVtxGenerator",
                            useCF = cms.untracked.bool(True),
                            signalLabel = cms.InputTag("generator"),
                            mixLabel = cms.InputTag("mix","generator")
                            )
