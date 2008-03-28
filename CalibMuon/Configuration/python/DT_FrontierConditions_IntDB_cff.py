import FWCore.ParameterSet.Config as cms

#
#DT calibrations
#For now, no t0 corrections are applied in the reconstruction if fake calibration is used
#(the replace should be moved to main cfg in order to avoid warning message)
from CalibMuon.DTCalibration.DT_Calibration_cff import *
maps_frontier.connect = 'frontier://cms_conditions_data/CMS_COND_20X_DT'
maps_frontier.toGet = cms.VPSet(cms.PSet(
    record = cms.string('DTT0Rcd'),
    tag = cms.string('t0Fake_20X_Sept15_mc')
), cms.PSet(
    record = cms.string('DTTtrigRcd'),
    tag = cms.string('ttrigFake_20X_July17_mc')
))

