import FWCore.ParameterSet.Config as cms

from SimGeneral.HepPDTESSource.pythiapdt_cfi import *
heavyIon = cms.EDProducer("GenHIEventProducer",
                          src = cms.InputTag("mix","generatorSmeared"),
                          )



