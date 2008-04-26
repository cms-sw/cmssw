import FWCore.ParameterSet.Config as cms

AlignmentMonitorMuonResidualsDefaults = cms.PSet(
    xresid_bins = cms.uint32(200), xresid_low = cms.double(-5.0), xresid_high = cms.double(5.0),
    xmean_bins = cms.uint32(400),  xmean_low = cms.double(-1.0),  xmean_high = cms.double(1.0),
    xstdev_bins = cms.uint32(300), xstdev_low = cms.double(0.0),  xstdev_high = cms.double(3.0),
    yresid_bins = cms.uint32(200), yresid_low = cms.double(-5.0), yresid_high = cms.double(5.0),
    ymean_bins = cms.uint32(400),  ymean_low = cms.double(-1.0),  ymean_high = cms.double(1.0),
    ystdev_bins = cms.uint32(300), ystdev_low = cms.double(0.0),  ystdev_high = cms.double(3.0),
)
