import FWCore.ParameterSet.Config as cms

# Put here the modules you want the cfg file to use,
# then include this file in your cfg file.
# i.e. in Validator.cfg replace 'module demo = Validator {} '
# with 'include "anlyzerDir/Validator/data/Validator.cfi" '.
# (Remember that filenames are case sensitive.)
TrackerOfflineValidation = cms.EDFilter("TrackerOfflineValidation",
    Tracks = cms.InputTag("TrackRefitter"),
    TH1NormResModules = cms.PSet(
        xmin = cms.double(-3.0),
        Nbinx = cms.int32(90),
        xmax = cms.double(3.0)
    ),
    trajectoryInput = cms.string('TrackRefitter'),
    TH1ResModules = cms.PSet(
        xmin = cms.double(-3.0),
        Nbinx = cms.int32(1000),
        xmax = cms.double(3.0)
    )
)


