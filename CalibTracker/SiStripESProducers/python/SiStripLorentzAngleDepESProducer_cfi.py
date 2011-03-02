# -*- coding: utf-8 -*-
import FWCore.ParameterSet.Config as cms

siStripLorentzAngleDepESProducer = cms.ESProducer("SiStripLorentzAngleDepESProducer",
   
     LatencyRecord =   cms.PSet(
            record = cms.string('SiStripLatencyRcd'),
            label = cms.untracked.string('Latency')
            ),
	LorentzAnglePeakMode = cms.PSet(
            record = cms.string('SiStripLorentzAngleRcd'),
            label = cms.untracked.string('LA1')
            ),
        LorentzAngleDeconvMode = cms.PSet(
            record = cms.string('SiStripLorentzAngleRcd'),
            label = cms.untracked.string('LA2')
            )
        
   )


