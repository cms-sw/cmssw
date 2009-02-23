import FWCore.ParameterSet.Config as cms
bscTrigger=cms.EDProducer("BSCTrigger",
                          bitNumbers=cms.vuint32(36,37,38,39,40,41),
                          bitNames=cms.vstring('L1TT_BscHaloPlusZInner', 
                                               'L1TT_BscHaloMinusZInner', 
                                               'L1TT_BscHaloPlusZOuter', 
                                               'L1TT_BscHaloMinusZOuter',
                                               'L1TT_BscMinBiasInner',
                                               'L1TT_BscMinBiasOuter'),
                          coincidence=cms.double(72.85),
                          resolution=cms.double(3.),
                          minbiasInnerMin=cms.int32(1),
                          minbiasOuterMin=cms.int32(1),
			  theHits=cms.InputTag('g4SimHits','BSCHits')
			  )		  

