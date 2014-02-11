import FWCore.ParameterSet.Config as cms
 
from IOMC.EventVertexGenerators.VtxSmearedParameters_cfi import *
VtxSmeared = cms.EDProducer("MixEvtVtxGenerator",
                            signalLabel = cms.InputTag("hiSignal"), 
                            heavyIonLabel = cms.InputTag("generator")
                            )
