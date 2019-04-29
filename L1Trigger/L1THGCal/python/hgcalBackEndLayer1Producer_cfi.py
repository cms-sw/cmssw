from __future__ import absolute_import
import FWCore.ParameterSet.Config as cms

import SimCalorimetry.HGCalSimProducers.hgcalDigitizer_cfi as digiparam
import RecoLocalCalo.HGCalRecProducers.HGCalUncalibRecHit_cfi as recoparam
import RecoLocalCalo.HGCalRecProducers.HGCalRecHit_cfi as recocalibparam 
from . import hgcalLayersCalibrationCoefficients_cfi as layercalibparam

C2d_parValues = cms.PSet( clusterType = cms.string('dummyC2d'),
                          calibSF_cluster=cms.double(1.),
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
