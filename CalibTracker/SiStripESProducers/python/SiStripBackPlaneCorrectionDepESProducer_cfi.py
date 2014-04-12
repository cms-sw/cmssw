# -*- coding: utf-8 -*-
import FWCore.ParameterSet.Config as cms

siStripBackPlaneCorrectionDepESProducer = cms.ESProducer("SiStripBackPlaneCorrectionDepESProducer",
   
     LatencyRecord =   cms.PSet(
            record = cms.string('SiStripLatencyRcd'),
            label = cms.untracked.string('')
            ),
	BackPlaneCorrectionPeakMode = cms.PSet(
            record = cms.string('SiStripBackPlaneCorrectionRcd'),
            label = cms.untracked.string('peak')
            ),
        BackPlaneCorrectionDeconvMode = cms.PSet(
            record = cms.string('SiStripBackPlaneCorrectionRcd'),
            label = cms.untracked.string('deconvolution')
            )
   )
