import FWCore.ParameterSet.Config as cms

spqhistory = cms.EDAnalyzer("SiPixelQualityHistory",
                            runProcess = cms.bool(False),
                            granularityMode=cms.untracked.uint32(2),  # 0 = Summary , 1= Module , 2= ROC
                            monitoredSiPixelQuality = cms.VPSet(
    cms.PSet( name = cms.string("Pixel"), spqLabel = cms.string(""))  # name= used in histos, ssqLabel= label of SiPixelQuality object
    )

                              )
