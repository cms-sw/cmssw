import FWCore.ParameterSet.Config as cms

from CalibTracker.SiPixelLorentzAngle.SiPixelLorentzAngleHLTFilter_cfi import *
seqALCARECOSiPixelLorentzAngle = cms.Sequence(SiPixelLorentzAngleHLTFilter)

