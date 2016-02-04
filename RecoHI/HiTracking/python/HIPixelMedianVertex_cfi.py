import FWCore.ParameterSet.Config as cms

hiPixelMedianVertex = cms.EDProducer("HIPixelMedianVtxProducer",
    TrackCollection = cms.InputTag('hiPixel3ProtoTracks'),
    PtMin = cms.double(0.075),           # selection already made in pixel track filter
    PeakFindThreshold = cms.uint32(100), # ntrack threshold for fitting vz around peak rather than median
    PeakFindMaxZ = cms.double(30.0),     # vz range for peak finding histogram
    PeakFindBinsPerCm = cms.int32(10),   # binning of histogram
    FitThreshold = cms.int32(5),         # minimum number of entries to attempt vz-fit
    FitMaxZ = cms.double(0.1),           # vz range (around max) of peak fitting histogram
    FitBinsPerCm = cms.int32(500)        # binning of histogram
)


