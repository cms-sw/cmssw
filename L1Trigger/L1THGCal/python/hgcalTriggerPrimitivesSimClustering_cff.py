import FWCore.ParameterSet.Config as cms

from L1Trigger.L1HGCal.hgcalTriggerPrimitives_cff import *

cluster_algo_all =  cms.PSet( AlgorithmName = cms.string('HGCalTriggerSimClusterBestChoice'),
                              FECodec = process.l1tHGCalTriggerPrimitiveDigiProducer.FECodec ,
                              HGCalEESensitive_tag = cms.string('HGCalEESensitive'),
                              HGCalHESiliconSensitive_tag = cms.string('HGCalHESiliconSensitive'),

                              calib_parameters = process.l1tHGCalTriggerPrimitiveDigiProducer.BEConfiguration.algorithms[0].calib_parameters,
			      simcollection=cms.InputTag("mix","MergedCaloTruth",""),
			      simhitsee = cms.InputTag("g4SimHits","HGCHitsEE",""),
			      simhitsfh = cms.InputTag("g4SimHits","HGCHitsHEfront",""),
		)


#process.hgcl1tpg_step = cms.Path(process.hgcalTriggerPrimitives)
