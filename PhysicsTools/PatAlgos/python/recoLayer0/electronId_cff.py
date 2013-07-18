import FWCore.ParameterSet.Config as cms

from RecoEgamma.ElectronIdentification.electronIdSequence_cff import *

patElectronId = cms.Sequence(
    eidRobustHighEnergy
)
