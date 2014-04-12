import FWCore.ParameterSet.Config as cms

from CalibTracker.SiStripLorentzAngle.Tree_RAW_cff import *
from CalibTracker.SiStripLorentzAngle.redigi_cff import *
siStripDigis.ProductLabel = 'SiStripDigiToRaw'

#Schedule
schedule = cms.Schedule( redigi_step, reconstruction_step, filter_refit_ntuplize_step )
