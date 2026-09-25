import FWCore.ParameterSet.Config as cms
from Configuration.ProcessModifiers.ticl_barrel_cff import ticl_barrel

# inclusive ECAL+HCAL+HGCal cells for the barrel layer clustering; attached
# only under the ticl_barrel process modifier, since only that path consumes it
def _addHltTiclGeomWithBarrel(process):
    process.hltTiclGeomWithBarrelESProducer = cms.ESProducer('TICLGeomESProducer@alpaka',
        detectors = cms.vstring('ECAL', 'HCAL', 'HGCal'),
        appendToDataLabel = cms.string('withBarrel')
    )

addHltTiclGeomWithBarrel = ticl_barrel.makeProcessModifier(_addHltTiclGeomWithBarrel)
