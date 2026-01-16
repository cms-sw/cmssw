import FWCore.ParameterSet.Config as cms
from .CE_E_Front_300um_cfi import generator

_pgunPSet = generator.PGunParameters

_pgunPSet.ControlledByREta = cms.bool(True)
_pgunPSet.MinEta = cms.double(-1.479)
_pgunPSet.MaxEta = cms.double(1.479)
_pgunPSet.RMin = cms.double(128.4)
_pgunPSet.RMax = cms.double(128.5)
_pgunPSet.ZMin = cms.double(-230)
_pgunPSet.ZMax = cms.double(230)

generator.PGunParameters = _pgunPSet
