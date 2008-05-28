import FWCore.ParameterSet.Config as cms

from SimGeneral.HepPDTESSource.pythiapdt_cfi import *
#The Tracker geometry ESProducer's (two producers, one for an aligned, 
# one for a misaligned geometry, identical by default
from Geometry.TrackerNumberingBuilder.trackerNumberingGeometry_cfi import *
from FastSimulation.Configuration.trackerGeometry_cfi import *
from FastSimulation.Configuration.TrackerRecoGeometryESProducer_cfi import *
from FastSimulation.TrackerSetup.TrackerInteractionGeometryESProducer_cfi import *
#The Magnetic Field ESProducer's
from MagneticField.Engine.volumeBasedMagneticField_cfi import *
from FastSimulation.ParticlePropagator.MagneticFieldMapESProducer_cfi import *
# The Calo geometry service model
from Geometry.CaloEventSetup.CaloGeometry_cff import *
from Geometry.CaloEventSetup.CaloTopology_cfi import *
from Geometry.CaloEventSetup.EcalTrigTowerConstituents_cfi import *
from Geometry.EcalMapping.EcalMappingRecord_cfi import *
# The muon geometry
from RecoMuon.DetLayers.muonDetLayerGeometry_cfi import *
# The condDB setup (the global tag refers to DevDB, IntDB or ProDB whenever needed)
from Configuration.StandardSequences.FrontierConditions_GlobalTag_cfi import *
hcal_db_producer = cms.ESProducer("HcalDbProducer",
    dump = cms.untracked.vstring(''),
    file = cms.untracked.string('')
)

es_hardcode = cms.ESSource("HcalHardcodeCalibrations",
    toGet = cms.untracked.vstring('GainWidths', 
        'channelQuality', 
        'ZSThresholds')
)



