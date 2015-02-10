import FWCore.ParameterSet.Config as cms

SiStripClusterChargeCutNone = cms.PSet(
    value     = cms.double(-1.0)
)
  
SiStripClusterChargeCutLoose = cms.PSet(
    value     = cms.double(1724.0) 
)

SiStripClusterChargeCutTight = cms.PSet(
    value     = cms.double(2069.0) 
)

