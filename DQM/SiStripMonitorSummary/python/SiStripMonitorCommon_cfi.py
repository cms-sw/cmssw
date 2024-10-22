import FWCore.ParameterSet.Config as cms

MonitorSiStrip_PSet = cms.PSet(
     MonitorSiStripPedestal     = cms.bool(False),
     MonitorSiStripNoise        = cms.bool(True),
     MonitorSiStripQuality      = cms.bool(True),
     MonitorSiStripApvGain      = cms.bool(False),
     MonitorSiStripLorentzAngle = cms.bool(False),
     MonitorSiStripBackPlaneCorrection = cms.bool(False),
     MonitorSiStripCabling      = cms.bool(False),
     MonitorSiStripLowThreshold = cms.bool(False),
     MonitorSiStripHighThreshold= cms.bool(False),
     )

