import FWCore.ParameterSet.Config as cms

SiStripClusterChargeCutNone = cms.PSet(
    value     = cms.double(-1.0)
)
  
SiStripClusterChargeCutTiny = cms.PSet(
    value     = cms.double(800.0)
)

SiStripClusterChargeCutLoose = cms.PSet(
    value     = cms.double(1620.0) 
)

SiStripClusterChargeCutTight = cms.PSet(
    value     = cms.double(1945.0) 
)

from Configuration.Eras.Modifier_trackingLowPU_cff import trackingLowPU
trackingLowPU.toReplaceWith(SiStripClusterChargeCutLoose, SiStripClusterChargeCutTiny)
trackingLowPU.toReplaceWith(SiStripClusterChargeCutTight, SiStripClusterChargeCutTiny)
