import FWCore.ParameterSet.Config as cms

from RecoEgamma.ElectronIdentification.electronIdSequence_cff import *
eidRobustHighEnergy = eidCutBasedExt.clone()
eidRobustHighEnergy.robustEleIDCuts.barrel = [0.050, 0.011, 0.090, 0.005]
eidRobustHighEnergy.robustEleIDCuts.endcap = [0.100, 0.0275, 0.090, 0.007]

patElectronId = cms.Sequence(
    eidRobustHighEnergy
)
