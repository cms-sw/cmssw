import FWCore.ParameterSet.Config as cms

siPixelDigisMorphed = cms.EDProducer(
    "SiPixelDigiMorphing",
    src = cms.InputTag('siPixelDigis'),
    nrows = cms.int32(160),
    ncols = cms.int32(416),
    nrocs = cms.int32(8),
    iters = cms.int32(1),
    kernel1 = cms.vint32(7, 7, 7),
    kernel2 = cms.vint32(2, 7, 2),
    fakeAdc = cms.uint32(100) # note, unit is 10e-
)


