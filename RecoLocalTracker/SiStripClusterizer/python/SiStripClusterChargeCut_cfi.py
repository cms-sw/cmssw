import FWCore.ParameterSet.Config as cms

SiStripClusterChargeCutNone = cms.PSet(
    value     = cms.untracked.double(-1.0)
)
  
SiStripClusterChargeCutLoose = cms.PSet(
    value     = cms.untracked.double(1724.0) 
)

SiStripClusterChargeCutStrict = cms.PSet(
    value     = cms.untracked.double(2069.0) 
)

