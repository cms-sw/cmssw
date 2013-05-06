import FWCore.ParameterSet.Config as cms
 
highLumiFilter = cms.EDFilter("HighLumiFilter",
                              lumiTag = cms.InputTag('lumiProducer'),
                              minLumi = cms.double(1.08)
                              )
