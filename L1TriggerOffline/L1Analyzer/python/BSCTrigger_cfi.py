import FWCore.ParameterSet.Config as cms
bscTrigger=cms.EDProducer("BSCTrigger",
                          bitNumbers=cms.vuint32(36,37,38,39,40,41),
                          bitPrescales=cms.vuint32(1,1,1,1,1,1),
                          bitNames=cms.vstring('BSC_H_IP','BSC_H_IM','BSC_H_OP','BSC_H_OM','BSC_MB_I','BSC_MB_O'),
                          coincidence=cms.double(72.85),
			  resolution=cms.double(3.),
			  minbiasInnerMin=cms.int32(1),
			  minbiasOuterMin=cms.int32(1)
			  )

