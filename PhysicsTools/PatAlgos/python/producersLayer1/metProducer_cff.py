import FWCore.ParameterSet.Config as cms

from PhysicsTools.PatAlgos.recoLayer0.metCorrections_cff import *
from PhysicsTools.PatAlgos.producersLayer1.metProducer_cfi import *

## for scheduled mode
makePatMETs = cms.Sequence(
    patMETCorrections *
    patMETs
    )
