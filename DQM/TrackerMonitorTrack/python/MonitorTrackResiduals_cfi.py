import FWCore.ParameterSet.Config as cms

# MonitorTrackResiduals
MonitorTrackResiduals = cms.EDAnalyzer("MonitorTrackResiduals",
    OutputMEsInRootFile = cms.bool(False),
    # should histogramms on module level be booked and filled?
    Mod_On = cms.bool(True),
    trajectoryInput = cms.string('TrackRefitter'),
    OutputFileName = cms.string('test_monitortracks.root'),
    # bining and range for absolute and normalized residual histogramms
    TH1ResModules = cms.PSet(
        xmin = cms.double(-0.05),   # native unit in CMS is [cm], so these are 500um
        Nbinx = cms.int32(100),
        xmax = cms.double(0.05)
    ),
    TH1NormResModules = cms.PSet(
        xmin = cms.double(-5.0),
        Nbinx = cms.int32(100),
        xmax = cms.double(5.0)
    ),
    # input for Tracks and Trajectories, should be TrackRefitter
    # or similar
    Tracks = cms.InputTag("TrackRefitter"),
    # should all MEs be reset after each run?
    ResetAfterRun = cms.bool(True)
)


