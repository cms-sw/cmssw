import FWCore.ParameterSet.Config as cms


from Alignment.OfflineValidation.TrackerOfflineValidation_cfi import *


# do the parameter setting before cloning, so the clone gets these values

TrackerOfflineValidation.TH1NormXprimeResStripModules.Nbinx = 120
TrackerOfflineValidation.TH1NormXprimeResStripModules.xmin = -3.0
TrackerOfflineValidation.TH1NormXprimeResStripModules.xmax = 3.0

TrackerOfflineValidation.TH1XprimeResStripModules.Nbinx = 2000
TrackerOfflineValidation.TH1XprimeResStripModules.xmin = -0.5
TrackerOfflineValidation.TH1XprimeResStripModules.xmax = 0.5

TrackerOfflineValidation.TH1NormYResStripModules.Nbinx = 120
TrackerOfflineValidation.TH1NormYResStripModules.xmin = -3.0
TrackerOfflineValidation.TH1NormYResStripModules.xmax = 3.0

TrackerOfflineValidation.TH1YResStripModules.Nbinx = 2000
TrackerOfflineValidation.TH1YResStripModules.xmin = -10.0
TrackerOfflineValidation.TH1YResStripModules.xmax = 10.0

TrackerOfflineValidation.TH1NormXprimeResPixelModules.Nbinx = 120
TrackerOfflineValidation.TH1NormXprimeResPixelModules.xmin = -3.0
TrackerOfflineValidation.TH1NormXprimeResPixelModules.xmax = 3.0

TrackerOfflineValidation.TH1XprimeResPixelModules.Nbinx = 2000
TrackerOfflineValidation.TH1XprimeResPixelModules.xmin = -0.5
TrackerOfflineValidation.TH1XprimeResPixelModules.xmax = 0.5

TrackerOfflineValidation.TH1NormYResPixelModules .Nbinx = 120
TrackerOfflineValidation.TH1NormYResPixelModules .xmin = -3.0
TrackerOfflineValidation.TH1NormYResPixelModules .xmax = 3.0

TrackerOfflineValidation.TH1YResPixelModules.Nbinx = 2000
TrackerOfflineValidation.TH1YResPixelModules.xmin = -0.5
TrackerOfflineValidation.TH1YResPixelModules.xmax = 0.5


TrackerOfflineValidationDqm = TrackerOfflineValidation.clone(
    useInDqmMode              = True,
    moduleDirectoryInOutput   = "AlCaReco/TkAl",
)

