import FWCore.ParameterSet.Config as cms

import SimCalorimetry.HGCalSimProducers.hgcalDigitizer_cfi as digiparam
import RecoLocalCalo.HGCalRecProducers.HGCalUncalibRecHit_cfi as recoparam
import RecoLocalCalo.HGCalRecProducers.HGCalRecHit_cfi as recocalibparam 
import hgcalLayersCalibrationCoefficients_cfi as layercalibparam

C2d_parValues = cms.PSet( #seeding_threshold = cms.double(5), # MipT
                          clusterType = cms.string('NNC2d'), # clustering type: dRC2d--> Geometric-dR clustering; NNC2d-->Nearest Neighbors clustering
                          #clustering_threshold = cms.double(2), # MipT, pas besoin
			  seeding_threshold_silicon = cms.double(5), # MipT
			  clustering_threshold_silicon = cms.double(2), # MipT
			  seeding_threshold_scintillator = cms.double(5), # MipT
			  clustering_threshold_scintillator = cms.double(2), # MipT
			  dR_cluster = cms.double(3.), # in cm
			  calibSF_cluster=cms.double(0.),
			  layerWeights = layercalibparam.TrgLayer_weights,
			  applyLayerCalibration = cms.bool(True)
			)

be_proc = cms.PSet( BeProcessorLayer1Name  = cms.string('HGCalBackendLayer1Processor'),
                    C2d_parameters = C2d_parValues.clone(),
		    #HGCalEESensitive_tag = cms.string('HGCalEESensitive'),
		    #HGCalHESiliconSensitive_tag = cms.string('HGCalHESiliconSensitive')
                  )


hgcalBackEndLayer1Producer = cms.EDProducer(
    "HGCalBackendLayer1Producer",
    bxCollection_be = cms.InputTag('hgcalConcentratorProducer:HGCalConcentratorProcessor'),
    Backendparam = be_proc.clone()
    )
