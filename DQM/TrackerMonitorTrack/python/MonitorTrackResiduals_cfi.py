import FWCore.ParameterSet.Config as cms

# MonitorTrackResiduals
MonitorTrackResiduals = cms.EDFilter("MonitorTrackResiduals",
    OutputMEsInRootFile = cms.bool(False),
    # should histogramms on module level be booked and filled?
    Mod_On = cms.bool(True),
    TH1NormResModules = cms.PSet(
        xmin = cms.double(-3.0),
        Nbinx = cms.int32(90),
        xmax = cms.double(3.0)
    ),
    trajectoryInput = cms.string('TrackRefitter'),
    OutputFileName = cms.string('test_monitortracks.root'),
    # input for Tracks and Trajectories, should be TrackRefitter
    # or similar
    Tracks = cms.InputTag("TrackRefitter"),
    # bining and range for absolute and normalized residual histogramms
    TH1ResModules = cms.PSet(
        xmin = cms.double(-3.0),
        Nbinx = cms.int32(90),
        xmax = cms.double(3.0)
    )
)


