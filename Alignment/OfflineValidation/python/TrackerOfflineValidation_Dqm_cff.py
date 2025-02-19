import FWCore.ParameterSet.Config as cms

##
## Set standard binning for the DMR histograms
##
from Alignment.OfflineValidation.TrackerOfflineValidationSummary_cfi import *

# do the parameter setting before cloning, so the clone gets these values
TrackerOfflineValidationSummary.TH1DmrXprimeStripModules.Nbinx = 50
TrackerOfflineValidationSummary.TH1DmrXprimeStripModules.xmin = -0.005
TrackerOfflineValidationSummary.TH1DmrXprimeStripModules.xmax = 0.005

TrackerOfflineValidationSummary.TH1DmrYprimeStripModules.Nbinx = 50
TrackerOfflineValidationSummary.TH1DmrYprimeStripModules.xmin = -0.005
TrackerOfflineValidationSummary.TH1DmrYprimeStripModules.xmax = 0.005

TrackerOfflineValidationSummary.TH1DmrXprimePixelModules.Nbinx = 50
TrackerOfflineValidationSummary.TH1DmrXprimePixelModules.xmin = -0.005
TrackerOfflineValidationSummary.TH1DmrXprimePixelModules.xmax = 0.005

TrackerOfflineValidationSummary.TH1DmrYprimePixelModules.Nbinx = 50
TrackerOfflineValidationSummary.TH1DmrYprimePixelModules.xmin = -0.005
TrackerOfflineValidationSummary.TH1DmrYprimePixelModules.xmax = 0.005

# First clone
TrackerOfflineValidationSummaryBinned = TrackerOfflineValidationSummary.clone()

##
## TrackerOfflineValidation (DQM mode)
##
from Alignment.OfflineValidation.TrackerOfflineValidation_Standalone_cff import TrackerOfflineValidationBinned
TrackerOfflineValidationDqm = TrackerOfflineValidationBinned.clone(
    useInDqmMode              = True,
    moduleDirectoryInOutput   = "Alignment/Tracker",
    Tracks =  'TrackRefitterForOfflineValidation',
)

##
## TrackerOfflineValidationSummary
##
TrackerOfflineValidationSummaryDqm = TrackerOfflineValidationSummaryBinned.clone(
    removeModuleLevelHists = True,
    minEntriesPerModuleForDmr = 100
)

##
## Output File Configuration
##
# DQM backend
from DQMServices.Core.DQM_cfg import *
# DQM file saver
DqmSaverTkAl = cms.EDAnalyzer("DQMFileSaver",
          convention=cms.untracked.string("Offline"),
          workflow=cms.untracked.string("/Cosmics/TkAl09-AlignmentSpecification_R000100000_R000100050_ValSkim-v1/ALCARECO"),   # /primaryDatasetName/WorkflowDescription/DataTier; Current Convention: Indicate run range (first and last run) in file name
	                                                                                                                       # WorkflowDescription should match namespace conventions, must have a year indicated by 2 digits in first word (aquisition era)) 
	  dirName=cms.untracked.string("."),
          saveByRun=cms.untracked.int32(-1),
	  saveAtJobEnd=cms.untracked.bool(True),                        
          forceRunNumber=cms.untracked.int32(100000)   # Current Convention: Take first processed run
)


##
## Sequence
##
seqTrackerOfflineValidationDqm = cms.Sequence(TrackerOfflineValidationDqm
                                              *TrackerOfflineValidationSummaryDqm
					      *DqmSaverTkAl
)
