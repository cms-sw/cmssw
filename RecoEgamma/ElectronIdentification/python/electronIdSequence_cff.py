import FWCore.ParameterSet.Config as cms

from RecoEgamma.ElectronIdentification.electronIdCutBasedExt_cfi import *

eidRobustLoose = eidCutBasedExt.clone()
eidRobustLoose.electronIDType = 'robust'
eidRobustLoose.electronQuality = 'loose'

eidRobustTight = eidCutBasedExt.clone()
eidRobustTight.electronIDType = 'robust'
eidRobustTight.electronQuality = 'tight'

eidRobustHighEnergy = eidCutBasedExt.clone()
eidRobustHighEnergy.electronIDType = 'robust'
eidRobustHighEnergy.electronQuality = 'highenergy'

eidLoose = eidCutBasedExt.clone()
eidLoose.electronIDType = 'classbased'
eidLoose.electronQuality = 'loose'

eidTight = eidCutBasedExt.clone()
eidTight.electronIDType = 'classbased'
eidTight.electronQuality = 'tight'

eIdSequence = cms.Sequence(eidRobustLoose+eidRobustTight+eidRobustHighEnergy+eidLoose+eidTight)
