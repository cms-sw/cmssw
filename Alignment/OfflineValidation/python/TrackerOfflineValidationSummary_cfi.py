import FWCore.ParameterSet.Config as cms

TrackerOfflineValidationSummary = cms.EDAnalyzer("TrackerOfflineValidationSummary",
   moduleDirectoryInOutput   = cms.string("Alignment/Tracker"),  # has to be the same as in TrackerOfflineValidation_Dqm_cff
   useFit                    = cms.bool(False),
   stripYDmrs                = cms.bool(False),  # should be the same as for stripYResiduals in TrackerOfflineValidation_Dqm_cff
   removeModuleLevelHists    = cms.bool(False),  # Remove module level hists after extracting the necessary information
   minEntriesPerModuleForDmr = cms.uint32(100),
   
   # DMR (distribution of median of residuals per module) of X coordinate (Strip)
   TH1DmrXprimeStripModules = cms.PSet(
         # width 2.0 um
#     Nbinx = cms.int32(500), xmin = cms.double(-0.05), xmax = cms.double(0.05)
     # width 0.5 um
    Nbinx = cms.int32(5000), xmin = cms.double(-0.05), xmax = cms.double(0.05)
   ),
   
   # DMR (distribution of median of residuals per module) of Y coordinate (Strip)
   TH1DmrYprimeStripModules = cms.PSet(
     Nbinx = cms.int32(200), xmin = cms.double(-0.05), xmax = cms.double(0.05)
   ),
   
   # DMR (distribution of median of residuals per module) of X coordinate (Pixel)
   TH1DmrXprimePixelModules = cms.PSet(
#     Nbinx = cms.int32(500), xmin = cms.double(-0.05), xmax = cms.double(0.05)
    Nbinx = cms.int32(5000), xmin = cms.double(-0.05), xmax = cms.double(0.05)
   ),
   
   # DMR (distribution of median of residuals per module) of Y coordinate (Pixel)
   TH1DmrYprimePixelModules = cms.PSet(
#     Nbinx = cms.int32(200), xmin = cms.double(-0.05), xmax = cms.double(0.05)
    Nbinx = cms.int32(5000), xmin = cms.double(-0.05), xmax = cms.double(0.05)
   )
   
)
