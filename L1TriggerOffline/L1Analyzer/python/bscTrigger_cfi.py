import FWCore.ParameterSet.Config as cms
bscTrigger=cms.EDProducer("BSCTrigger",
                          bitNumbers=cms.vuint32(36,37,38,39,40,41),
                          bitNames=cms.vstring('L1Tech_BscHaloPlusZInner', 
                                               'L1Tech_BscHaloMinusZInner', 
                                               'L1Tech_BscHaloPlusZOuter', 
                                               'L1Tech_BscHaloMinusZOuter',
                                               'L1Tech_BscMinBiasInner',
                                               'L1Tech_BscMinBiasOuter'),
                          coincidence=cms.double(72.85),
                          resolution=cms.double(3.),
                          minbiasInnerMin=cms.int32(1),
                          minbiasOuterMin=cms.int32(1),
			  theHits=cms.InputTag('g4SimHits','BSCHits')
			  )		  

