import FWCore.ParameterSet.Config as cms

hltPhase2PixelFitterByHelixProjections = cms.EDProducer("PixelFitterByHelixProjectionsProducer",
    scaleErrorsForBPix1 = cms.bool(False),
    scaleFactor = cms.double(0.65)
)
