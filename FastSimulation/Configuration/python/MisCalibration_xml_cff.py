import FWCore.ParameterSet.Config as cms

from CalibCalorimetry.Configuration.Ecal_FakeConditions_cff import *
CaloMiscalibTools = cms.ESSource("CaloMiscalibTools",
    fileNameEndcap = cms.untracked.string(''),
    fileNameBarrel = cms.untracked.string('miscalib_barrel_0.05.xml')
)

es_prefer_CaloMiscalibTools = cms.ESPrefer("CaloMiscalibTools")
caloRecHits.RecHitsFactory.doMiscalib = True

