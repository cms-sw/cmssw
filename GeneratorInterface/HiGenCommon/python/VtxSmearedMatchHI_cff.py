import FWCore.ParameterSet.Config as cms
 
from IOMC.EventVertexGenerators.VtxSmearedParameters_cfi import *
VtxSmeared = cms.EDProducer("MixEvtVtxGenerator",
                            useCF = cms.untracked.bool(True),
                            mixLabel = cms.InputTag("mixGen"), 
                            signalLabel = cms.InputTag("NONE"), 
                            heavyIonLabel = cms.InputTag("NONE")
                            )
