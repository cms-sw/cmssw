import FWCore.ParameterSet.Config as cms

# MonitorTrackResiduals
MonitorTrackResiduals = cms.EDFilter("MonitorTrackResiduals",
    OutputMEsInRootFile = cms.bool(False),
    # should histogramms on module level be booked and filled?
    Mod_On = cms.bool(True),
    trajectoryInput = cms.string('TrackRefitter'),
    OutputFileName = cms.string('test_monitortracks.root'),
    # bining and range for absolute and normalized residual histogramms
    TH1ResModules = cms.PSet(
        xmin = cms.double(-2.0),
        Nbinx = cms.int32(120),
        xmax = cms.double(2.0)
    ),
    TH1NormResModules = cms.PSet(
        xmin = cms.double(-3.0),
        Nbinx = cms.int32(120),
        xmax = cms.double(3.0)
    ),
    # input for Tracks and Trajectories, should be TrackRefitter
    # or similar
    Tracks = cms.InputTag("TrackRefitter"),
    # should all MEs be reset after each run?
    ResetAfterRun = cms.bool(True)
)


