import FWCore.ParameterSet.Config as cms

from PhysicsTools.PatAlgos.recoLayer0.jetCorrections_cff import *
from PhysicsTools.PatAlgos.recoLayer0.metCorrections_cff import *

## for scheduled mode
patJetMETCorrections = cms.Sequence(patJetCorrections*patMETCorrections)


