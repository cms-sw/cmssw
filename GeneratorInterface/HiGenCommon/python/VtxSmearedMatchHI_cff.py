import FWCore.ParameterSet.Config as cms
 
from IOMC.EventVertexGenerators.VtxSmearedParameters_cfi import *
VertexSmearingParameters = cms.PSet(
                            vertexGeneratorType = cms.string("MixEvtVtxGenerator"),
                            useCF = cms.untracked.bool(True),
                            signalLabel = cms.InputTag("generator"),
                            mixLabel = cms.InputTag("mix","generator")
                            )
