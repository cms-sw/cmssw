from __future__ import absolute_import
import FWCore.ParameterSet.Config as cms

import SimCalorimetry.HGCalSimProducers.hgcalDigitizer_cfi as digiparam
import RecoLocalCalo.HGCalRecProducers.HGCalUncalibRecHit_cfi as recoparam
import RecoLocalCalo.HGCalRecProducers.HGCalRecHit_cfi as recocalibparam
from . import hgcalLayersCalibrationCoefficients_cfi as layercalibparam

from Configuration.Eras.Modifier_phase2_hgcalV9_cff import phase2_hgcalV9

c2d_calib_pset = cms.PSet(calibSF_cluster=cms.double(1.),
                          layerWeights=layercalibparam.TrgLayer_weights,
                          applyLayerCalibration=cms.bool(True))

c2d_thresholds_pset = cms.PSet(seeding_threshold_silicon=cms.double(5.),
                               seeding_threshold_scintillator=cms.double(5.),
                               clustering_threshold_silicon=cms.double(2.),
                               clustering_threshold_scintillator=cms.double(2.))

# V9 samples have a different defintiion of the dEdx calibrations. To account for it
# we reascale the thresholds for the clustering
# (see https://indico.cern.ch/event/806845/contributions/3359859/attachments/1815187/2966402/19-03-20_EGPerf_HGCBE.pdf
# for more details)
phase2_hgcalV9.toModify(c2d_thresholds_pset,
                        seeding_threshold_silicon=3.75,
                        seeding_threshold_scintillator=3.75,
                        clustering_threshold_silicon=1.5,
                        clustering_threshold_scintillator=1.5,
                        )

# we still don't have layer calibrations for V9 geometry. Switching this off we
# use the dEdx calibrated energy of the TCs
phase2_hgcalV9.toModify(c2d_calib_pset,
                        applyLayerCalibration=False
                        )


dummy_C2d_params = cms.PSet(c2d_calib_pset,
                            clusterType=cms.string('dummyC2d')
                            )


distance_C2d_params = cms.PSet(c2d_calib_pset,
                               c2d_thresholds_pset,
                               clusterType=cms.string('dRC2d'),
                               dR_cluster=cms.double(6.),
                               )

topological_C2d_params = cms.PSet(c2d_calib_pset,
                                  c2d_thresholds_pset,
                                  clusterType=cms.string('NNC2d'),
                                  )

constrTopological_C2d_params = cms.PSet(c2d_calib_pset,
                                        c2d_thresholds_pset,
                                        clusterType=cms.string('dRNNC2d'),
                                        dR_cluster=cms.double(6.),
                                        )


be_proc = cms.PSet(ProcessorName  = cms.string('HGCalBackendLayer1Processor2DClustering'),
                   C2d_parameters = dummy_C2d_params.clone()
                   )

hgcalBackEndLayer1Producer = cms.EDProducer(
    "HGCalBackendLayer1Producer",
    InputTriggerCells = cms.InputTag('hgcalConcentratorProducer:HGCalConcentratorProcessorSelection'),
    ProcessorParameters = be_proc.clone()
    )
