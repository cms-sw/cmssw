import FWCore.ParameterSet.Config as cms                                          

filterOutLowPt = cms.EDFilter("FilterOutLowPt",
                              applyfilter = cms.untracked.bool(True),
                              src =  cms.untracked.InputTag("ALCARECOTkAlMinBias"),
                              debugOn = cms.untracked.bool(False),
                              numtrack = cms.untracked.uint32(0),
                              thresh = cms.untracked.int32(1),
                              ptmin  = cms.untracked.double(0.),
                              runControl = cms.untracked.bool(False),
                              runControlNumber = cms.untracked.vuint32(1),
                              )
