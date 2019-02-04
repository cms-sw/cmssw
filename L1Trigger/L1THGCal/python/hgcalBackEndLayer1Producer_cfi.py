import FWCore.ParameterSet.Config as cms

import SimCalorimetry.HGCalSimProducers.hgcalDigitizer_cfi as digiparam
import RecoLocalCalo.HGCalRecProducers.HGCalUncalibRecHit_cfi as recoparam
import RecoLocalCalo.HGCalRecProducers.HGCalRecHit_cfi as recocalibparam 
import hgcalLayersCalibrationCoefficients_cfi as layercalibparam

C2d_parValues = cms.PSet( clusterType = cms.string('dRNNC2d'), # clustering type: dRC2d--> Geometric-dR clustering; NNC2d-->Nearest Neighbors clustering
                          seeding_threshold_silicon = cms.double(5), # MipT
                          seeding_threshold_scintillator = cms.double(5), # MipT
                          clustering_threshold_silicon = cms.double(2), # MipT
                          clustering_threshold_scintillator = cms.double(2), # MipT
                          dR_cluster = cms.double(6.), # in cm
                          calibSF_cluster=cms.double(0.),
                          layerWeights = layercalibparam.TrgLayer_weights,
                          applyLayerCalibration = cms.bool(True)
            )

be_proc = cms.PSet( ProcessorName  = cms.string('HGCalBackendLayer1Processor2DClustering'),
                    C2d_parameters = C2d_parValues.clone()
                  )

hgcalBackEndLayer1Producer = cms.EDProducer(
    "HGCalBackendLayer1Producer",
    InputTriggerCells = cms.InputTag('hgcalConcentratorProducer:HGCalConcentratorProcessorSelection'),
    ProcessorParameters = be_proc.clone()
    )
