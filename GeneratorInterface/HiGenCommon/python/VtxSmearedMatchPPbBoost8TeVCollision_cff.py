import FWCore.ParameterSet.Config as cms
 
from IOMC.EventVertexGenerators.VtxSmearedParameters_cfi import *
VtxSmeared = cms.EDProducer("MixBoostEvtVtxGenerator",
                            signalLabel = cms.InputTag("hiSignal"), 
                            heavyIonLabel = cms.InputTag("generator"),
                            Beta=cms.double(-0.434)                          
                            )
