import FWCore.ParameterSet.Config as cms

from Alignment.OfflineValidation.trackerOfflineValidationSummary_cfi import trackerOfflineValidationSummary as _trackerOfflineValidationSummary

TrackerOfflineValidationSummary =  _trackerOfflineValidationSummary.clone(
    moduleDirectoryInOutput   = "Alignment/Tracker",  # has to be the same as in TrackerOfflineValidation_Dqm_cff
    useFit                    = False,
    stripYDmrs                = False,  # should be the same as for stripYResiduals in TrackerOfflineValidation_Dqm_cff
    minEntriesPerModuleForDmr = 100,

    # DMR (distribution of median of residuals per module) of X coordinate (Strip)
    # width 2.0 um
    # Nbinx = cms.int32(500), xmin = cms.double(-0.05), xmax = cms.double(0.05)
    # width 0.5 um
    TH1DmrXprimeStripModules = dict(Nbinx = 5000, xmin = -0.05, xmax = 0.05),

    # DMR (distribution of median of residuals per module) of Y coordinate (Strip)
    TH1DmrYprimeStripModules = dict(Nbinx = 200, xmin = -0.05, xmax = 0.05),

    # DMR (distribution of median of residuals per module) of X coordinate (Pixel)
    # Nbinx = cms.int32(500), xmin = cms.double(-0.05), xmax = cms.double(0.05)
    TH1DmrXprimePixelModules = dict(Nbinx = 5000, xmin = -0.05, xmax = 0.05),

    # DMR (distribution of median of residuals per module) of Y coordinate (Pixel)
    # Nbinx = cms.int32(200), xmin = cms.double(-0.05), xmax = cms.double(0.05)
    TH1DmrYprimePixelModules = dict(Nbinx = 5000, xmin = -0.05, xmax = 0.05)
)
