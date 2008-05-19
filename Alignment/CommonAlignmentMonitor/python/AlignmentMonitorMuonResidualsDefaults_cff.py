import FWCore.ParameterSet.Config as cms

AlignmentMonitorMuonResidualsDefaults = cms.PSet(
    tracker_hists = cms.bool(False),
    xresid_bins = cms.uint32(200), xresid_low = cms.double(-50.0), xresid_high = cms.double(50.0),
    xmean_bins = cms.uint32(400),  xmean_low = cms.double(-10.0),  xmean_high = cms.double(10.0),
    xstdev_bins = cms.uint32(300), xstdev_low = cms.double(0.0),  xstdev_high = cms.double(30.0),
    xerronmean_bins = cms.uint32(300), xerronmean_low = cms.double(0.0),  xerronmean_high = cms.double(1.0),
    yresid_bins = cms.uint32(200), yresid_low = cms.double(-50.0), yresid_high = cms.double(50.0),
    ymean_bins = cms.uint32(400),  ymean_low = cms.double(-10.0),  ymean_high = cms.double(10.0),
    ystdev_bins = cms.uint32(300), ystdev_low = cms.double(0.0),  ystdev_high = cms.double(30.0),
    yerronmean_bins = cms.uint32(300), yerronmean_low = cms.double(0.0),  yerronmean_high = cms.double(1.0),
)
