import FWCore.ParameterSet.Config as cms

AlignmentMonitorMuonResidualsDefaults = cms.PSet(
    yresid_bins = cms.uint32(200),
    ymean_low = cms.double(-1.0),
    outpath = cms.string('./'),
    collectorNJobs = cms.int32(0),
    ystdev_high = cms.double(3.0),
    xresid_high = cms.double(5.0),
    ystdev_bins = cms.uint32(300),
    xstdev_high = cms.double(3.0),
    collectorActive = cms.bool(False),
    collectorPath = cms.string('./'),
    yresid_high = cms.double(5.0),
    xstdev_bins = cms.uint32(300),
    xmean_high = cms.double(1.0),
    xmean_low = cms.double(-1.0),
    xstdev_low = cms.double(0.0),
    ymean_bins = cms.uint32(400),
    yresid_low = cms.double(-5.0),
    xresid_bins = cms.uint32(200),
    xresid_low = cms.double(-5.0),
    ystdev_low = cms.double(0.0),
    xmean_bins = cms.uint32(400),
    ymean_high = cms.double(1.0)
)

