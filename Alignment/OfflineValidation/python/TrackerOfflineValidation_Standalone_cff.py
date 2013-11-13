import FWCore.ParameterSet.Config as cms

##
## Set standard binning for the residual histograms in both, standalone and DQM mode
##
from Alignment.OfflineValidation.TrackerOfflineValidation_cfi import *

# do the parameter setting before cloning, so the clone gets these values
TrackerOfflineValidation.TH1NormXprimeResStripModules.Nbinx = 120
TrackerOfflineValidation.TH1NormXprimeResStripModules.xmin = -3.0
TrackerOfflineValidation.TH1NormXprimeResStripModules.xmax = 3.0

#TrackerOfflineValidation.TH1NormXResStripModules.Nbinx = 120
#TrackerOfflineValidation.TH1NormXResStripModules.xmin = -3.0
#TrackerOfflineValidation.TH1NormXResStripModules.xmax = 3.0

TrackerOfflineValidation.TH1XprimeResStripModules.Nbinx = 5000
TrackerOfflineValidation.TH1XprimeResStripModules.xmin = -0.05 #-0.5
TrackerOfflineValidation.TH1XprimeResStripModules.xmax = 0.05 #0.5

#TrackerOfflineValidation.TH1XResStripModules.Nbinx = 5000
#TrackerOfflineValidation.TH1XResStripModules.xmin = -0.5
#TrackerOfflineValidation.TH1XResStripModules.xmax = 0.5

TrackerOfflineValidation.TH1NormYResStripModules.Nbinx = 120
TrackerOfflineValidation.TH1NormYResStripModules.xmin = -3.0
TrackerOfflineValidation.TH1NormYResStripModules.xmax = 3.0

TrackerOfflineValidation.TH1YResStripModules.Nbinx = 5000
TrackerOfflineValidation.TH1YResStripModules.xmin = -11.0
TrackerOfflineValidation.TH1YResStripModules.xmax = 11.0

TrackerOfflineValidation.TH1NormXprimeResPixelModules.Nbinx = 120
TrackerOfflineValidation.TH1NormXprimeResPixelModules.xmin = -3.0
TrackerOfflineValidation.TH1NormXprimeResPixelModules.xmax = 3.0

#TrackerOfflineValidation.TH1NormXResPixelModules.Nbinx = 120
#TrackerOfflineValidation.TH1NormXResPixelModules.xmin = -3.0
#TrackerOfflineValidation.TH1NormXResPixelModules.xmax = 3.0

TrackerOfflineValidation.TH1XprimeResPixelModules.Nbinx = 5000
TrackerOfflineValidation.TH1XprimeResPixelModules.xmin = -0.05 #-0.5
TrackerOfflineValidation.TH1XprimeResPixelModules.xmax = 0.05 #0.5

#TrackerOfflineValidation.TH1XResPixelModules.Nbinx = 5000
#TrackerOfflineValidation.TH1XResPixelModules.xmin = -0.5
#TrackerOfflineValidation.TH1XResPixelModules.xmax = 0.5

TrackerOfflineValidation.TH1NormYResPixelModules.Nbinx = 120
TrackerOfflineValidation.TH1NormYResPixelModules.xmin = -3.0
TrackerOfflineValidation.TH1NormYResPixelModules.xmax = 3.0

TrackerOfflineValidation.TH1YResPixelModules.Nbinx = 5000
TrackerOfflineValidation.TH1YResPixelModules.xmin = -0.05 #-0.5
TrackerOfflineValidation.TH1YResPixelModules.xmax = 0.05 #0.5

TrackerOfflineValidation.TProfileXResStripModules.Nbinx = 34
TrackerOfflineValidation.TProfileXResStripModules.xmin = -1.02
TrackerOfflineValidation.TProfileXResStripModules.xmax = 1.02

TrackerOfflineValidation.TProfileXResPixelModules.Nbinx = 17
TrackerOfflineValidation.TProfileXResPixelModules.xmin = -1.02
TrackerOfflineValidation.TProfileXResPixelModules.xmax = 1.02

TrackerOfflineValidation.TProfileYResStripModules.Nbinx = 34
TrackerOfflineValidation.TProfileYResStripModules.xmin = -1.02
TrackerOfflineValidation.TProfileYResStripModules.xmax = 1.02

TrackerOfflineValidation.TProfileYResPixelModules.Nbinx = 17
TrackerOfflineValidation.TProfileYResPixelModules.xmin = -1.02
TrackerOfflineValidation.TProfileYResPixelModules.xmax = 1.02

# First clone contains the standard histogram binning for both, Standalone and DQMmode
TrackerOfflineValidationBinned = TrackerOfflineValidation.clone()

##
## TrackerOfflineValidation (standalone mode)
##
# Second clone
TrackerOfflineValidationStandalone = TrackerOfflineValidationBinned.clone(
    Tracks = 'TrackRefitterForOfflineValidation',
    moduleLevelHistsTransient = cms.bool(True),
    moduleLevelProfiles = cms.bool(False)
)

##
## Output File Configuration 
##
# use TFileService
from PhysicsTools.UtilAlgos.TFileService_cfi import *
TFileService = cms.Service("TFileService",
    fileName = cms.string('$TMPDIR/trackerOfflineValidation.root'),
    closeFileFast = cms.untracked.bool(True)
)

##
## Sequence
##
seqTrackerOfflineValidationStandalone = cms.Sequence(TrackerOfflineValidationStandalone)

