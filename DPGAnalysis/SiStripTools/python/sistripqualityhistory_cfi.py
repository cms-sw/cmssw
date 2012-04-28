import FWCore.ParameterSet.Config as cms

ssqhistory = cms.EDAnalyzer("SiStripQualityHistory",
                            eventProcessing = cms.bool(False),
                            granularityMode=cms.untracked.uint32(2),  # 1 = Module , 2= Fiber , 3= APV 
                            monitoredSiStripQuality = cms.VPSet(
#    cms.PSet( name = cms.string("Cabling"), ssqLabel = cms.string("onlyCabling")),  # name= used in histos, ssqLabel= label of SiStripQuality object
#    cms.PSet( name = cms.string("RunInfo"), ssqLabel = cms.string("CablingRunInfo")),
#    cms.PSet( name = cms.string("BadChannel"), ssqLabel = cms.string("BadChannel")),
#    cms.PSet( name = cms.string("BadFiber"), ssqLabel = cms.string("BadFiber"))
    )

                              )
