from __future__ import absolute_import
import FWCore.ParameterSet.Config as cms

import SimCalorimetry.HGCalSimProducers.hgcalDigitizer_cfi as digiparam
import RecoLocalCalo.HGCalRecProducers.HGCalUncalibRecHit_cfi as recoparam
import RecoLocalCalo.HGCalRecProducers.HGCalRecHit_cfi as recocalibparam
from . import hgcalLayersCalibrationCoefficients_cfi as layercalibparam

from Configuration.Eras.Modifier_phase2_hgcalV10_cff import phase2_hgcalV10

c2d_calib_pset = cms.PSet(calibSF_cluster=cms.double(1.),
                          layerWeights=layercalibparam.TrgLayer_weights,
                          applyLayerCalibration=cms.bool(True))

c2d_thresholds_pset = cms.PSet(seeding_threshold_silicon=cms.double(5.),
                               seeding_threshold_scintillator=cms.double(5.),
                               clustering_threshold_silicon=cms.double(2.),
                               clustering_threshold_scintillator=cms.double(2.))

# >= V9 samples have a different definition of the dEdx calibrations. To account for it
# we rescale the thresholds for the clustering
# (see https://indico.cern.ch/event/806845/contributions/3359859/attachments/1815187/2966402/19-03-20_EGPerf_HGCBE.pdf
# for more details)
phase2_hgcalV10.toModify(c2d_thresholds_pset,
                        seeding_threshold_silicon=3.75,
                        seeding_threshold_scintillator=3.75,
                        clustering_threshold_silicon=1.5,
                        clustering_threshold_scintillator=1.5,
                        )

# we still don't have layer calibrations for >=V9 geometry. Switching this off we
# use the dEdx calibrated energy of the TCs
phase2_hgcalV10.toModify(c2d_calib_pset,
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

ntcs_72links = [ 1,  4, 13, 13, 10, 10,  8,  8,  8,  7,  7,  6,  6,  6,  6,  6,  5,  5,  5,  5,  5,  5,  5,  4,  4,  4,  4,  4,  4,  4,  4,  4,  3,  2,  2,  2,  2,  2,  2,  2,  2,  1]
ntcs_120links = [ 2,  7, 27, 24, 19, 17, 16, 15, 14, 14, 13, 13, 13, 12, 12, 12, 11, 11, 11, 10, 10, 10, 10, 10,  9,  9, 10,  9,  9,  9,  8,  8,  7,  5,  3,  3,  3,  3,  3,  3,  3,  3]

phi_edges = [0.98901991,0.72722052,0.6981317,0.87266463,0.93084227,0.90175345,
0.87266463,0.90175345,0.95993109,0.95993109,0.93084227,0.93084227,
0.95993109,0.98901991,0.95993109,0.95993109,0.95993109,0.98901991,
0.98901991,0.95993109,0.95993109,0.98901991,0.98901991,0.98901991,
0.98901991,0.98901991,1.01810873,0.98901991,0.98901991,0.98901991,
0.98901991,0.98901991,0.98901991,0.98901991,1.04719755,1.04719755,
1.04719755,1.04719755,1.01810873,1.04719755,1.01810873,1.01810873]

truncation_params = cms.PSet(rozMin=cms.double(0.07587128),
        rozMax=cms.double(0.55508006),
        rozBins=cms.uint32(42),
        maxTcsPerBin=cms.vuint32(ntcs_120links),
        phiSectorEdges=cms.vdouble(phi_edges),
        doTruncation=cms.bool(True)
        )

truncation_paramsSA = cms.PSet(AlgoName=cms.string('HGCalStage1TruncationWrapper'),
        rozMin=cms.double(0.07587128),
        rozMax=cms.double(0.55508006),
        rozBins=cms.uint32(42),
        maxTcsPerBin=cms.vuint32(ntcs_120links),
        phiSectorEdges=cms.vdouble(phi_edges),
        doTruncation=cms.bool(True)
        )


layer1truncation_proc = cms.PSet(ProcessorName  = cms.string('HGCalBackendLayer1Processor'),
                   C2d_parameters = dummy_C2d_params.clone(),
                   truncation_parameters = truncation_params.clone()
                   )
stage1truncation_proc = cms.PSet(ProcessorName  = cms.string('HGCalBackendStage1Processor'),
                   truncation_parameters = truncation_paramsSA.clone()
                   )

be_proc = cms.PSet(ProcessorName  = cms.string('HGCalBackendLayer1Processor2DClustering'),
                   C2d_parameters = dummy_C2d_params.clone()
                   )

l1tHGCalBackEndLayer1Producer = cms.EDProducer(
    "HGCalBackendLayer1Producer",
    InputTriggerCells = cms.InputTag('l1tHGCalConcentratorProducer:HGCalConcentratorProcessorSelection'),
    ProcessorParameters = be_proc.clone()
    )

l1tHGCalBackEndStage1Producer = cms.EDProducer(
    "HGCalBackendStage1Producer",
    InputTriggerCells = cms.InputTag('l1tHGCalConcentratorProducer:HGCalConcentratorProcessorSelection'),
    C2d_parameters = dummy_C2d_params.clone(),
    ProcessorParameters = stage1truncation_proc.clone()
    )

l1tHGCalBackEndLayer1ProducerHFNose = l1tHGCalBackEndLayer1Producer.clone(
    InputTriggerCells = 'l1tHGCalConcentratorProducerHFNose:HGCalConcentratorProcessorSelection'
)
