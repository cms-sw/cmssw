import FWCore.ParameterSet.Config as cms

ssqhistory = cms.EDAnalyzer("SiStripQualityHistory",
                            runProcess = cms.bool(False),
                            granularityMode=cms.untracked.uint32(2),  # 0 = Module , 1= Fiber , 2= APV, 3=Strip 
                            monitoredSiStripQuality = cms.VPSet(
#    cms.PSet( name = cms.string("Cabling"), ssqLabel = cms.string("onlyCabling")),  # name= used in histos, ssqLabel= label of SiStripQuality object
#    cms.PSet( name = cms.string("RunInfo"), ssqLabel = cms.string("CablingRunInfo")),
#    cms.PSet( name = cms.string("BadChannel"), ssqLabel = cms.string("BadChannel")),
#    cms.PSet( name = cms.string("BadFiber"), ssqLabel = cms.string("BadFiber"))
    )

                              )
