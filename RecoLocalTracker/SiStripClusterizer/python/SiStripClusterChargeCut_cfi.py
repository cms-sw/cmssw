import FWCore.ParameterSet.Config as cms
from Configuration.StandardSequences.Eras import eras

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

eras.trackingLowPU.toReplaceWith(SiStripClusterChargeCutLoose, SiStripClusterChargeCutTiny)
eras.trackingLowPU.toReplaceWith(SiStripClusterChargeCutTight, SiStripClusterChargeCutTiny)
