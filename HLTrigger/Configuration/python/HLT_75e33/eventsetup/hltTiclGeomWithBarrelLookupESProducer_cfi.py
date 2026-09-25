import FWCore.ParameterSet.Config as cms
from Configuration.ProcessModifiers.ticl_barrel_cff import ticl_barrel

def _addHltTiclGeomWithBarrelLookup(process):
    process.hltTiclGeomWithBarrelLookupESProducer = cms.ESProducer('TICLGeomLookupESProducer@alpaka',
        src = cms.ESInputTag('', 'withBarrel'),
        appendToDataLabel = cms.string('withBarrel')
    )

addHltTiclGeomWithBarrelLookup = ticl_barrel.makeProcessModifier(_addHltTiclGeomWithBarrelLookup)
