import FWCore.ParameterSet.Config as cms

from CalibTracker.SiPixelQuality.SiPixelStatusProducer_cfi import *
siPixelStatus = cms.Sequence( siPixelStatusProducer )
