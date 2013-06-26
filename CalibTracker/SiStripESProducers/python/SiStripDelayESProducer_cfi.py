# -*- coding: utf-8 -*-
import FWCore.ParameterSet.Config as cms

siStripDelayESProducer = cms.ESProducer("SiStripDelayESProducer",
    ListOfRecordToMerge = cms.VPSet(
	cms.PSet(
	    Record = cms.string("SiStripBaseDelayRcd"),
	    Label = cms.string("baseDelay1"),
	    SumSign = cms.int32(1)
	),
	cms.PSet(
	    Record = cms.string("SiStripBaseDelayRcd"),
	    Label = cms.string("baseDelay2"),
	    SumSign = cms.int32(-1)
	)
    )
)
