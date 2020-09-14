import FWCore.ParameterSet.Config as cms

from RecoEgamma.ElectronIdentification.electronIdCutBasedExt_cfi import *

eidRobustLoose = eidCutBasedExt.clone(
    electronIDType  = 'robust',
    electronQuality = 'loose'
)
eidRobustTight = eidCutBasedExt.clone(
    electronIDType  = 'robust',
    electronQuality = 'tight'
)
eidRobustHighEnergy = eidCutBasedExt.clone(
    electronIDType  = 'robust',
    electronQuality = 'highenergy'
)
eidLoose = eidCutBasedExt.clone(
    electronIDType  = 'classbased',
    electronQuality = 'loose'
)
eidTight = eidCutBasedExt.clone(
    electronIDType  = 'classbased',
    electronQuality = 'tight'
)
eIdTask = cms.Task(eidRobustLoose,eidRobustTight,eidRobustHighEnergy,eidLoose,eidTight)
eIdSequence = cms.Sequence(eIdTask)
