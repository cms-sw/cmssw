import FWCore.ParameterSet.Config as cms

SiPixelFakeCPEGenericErrorParmESSource = cms.ESSource("SiPixelFakeCPEGenericErrorParmESSource",
    file = cms.FileInPath('RecoLocalTracker/SiPixelRecHits/data/residuals.dat'),
    version = cms.double(1)                                             
)


