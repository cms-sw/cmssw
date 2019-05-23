import FWCore.ParameterSet.Config as cms

from L1Trigger.L1THGCal.egammaIdentification import egamma_identification_drnn_cone, \
                                                    egamma_identification_drnn_dbscan, \
                                                    egamma_identification_histomax

from Configuration.Eras.Modifier_phase2_hgcalV9_cff import phase2_hgcalV9
from Configuration.Eras.Modifier_phase2_hgcalV10_cff import phase2_hgcalV10
from Configuration.Eras.Modifier_phase2_hfnose_cff import phase2_hfnose


binSums = cms.vuint32(13,               # 0
                      11, 11, 11,       # 1 - 3
                      9, 9, 9,          # 4 - 6
                      7, 7, 7, 7, 7, 7,  # 7 - 12
                      5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5,  # 13 - 27
                      3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3  # 28 - 41
                      )

binSumsNose = cms.vuint32(13,               # 0
                          13,               # 1
                          11, 11, 11,       # 2 - 4
                          9, 9, 9,          # 5 - 7
                          7, 7, 7, 7, 7, 7,  # 8 - 13
                          5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5,  # 14 - 28
                          3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3  # 29 - 42
)

EE_DR_GROUP = 7
FH_DR_GROUP = 6
BH_DR_GROUP = 12
MAX_LAYERS = 52


dr_layerbylayer = ([0] + # no layer 0
        [0.015]*EE_DR_GROUP + [0.020]*EE_DR_GROUP + [0.030]*EE_DR_GROUP + [0.040]*EE_DR_GROUP + # EM
        [0.040]*FH_DR_GROUP + [0.050]*FH_DR_GROUP + # FH
        [0.050]*BH_DR_GROUP) # BH


dr_layerbylayer_Bcoefficient = ([0] + # no layer 0
        [0.020]*EE_DR_GROUP + [0.020]*EE_DR_GROUP + [0.02]*EE_DR_GROUP + [0.020]*EE_DR_GROUP + # EM
        [0.020]*FH_DR_GROUP + [0.020]*FH_DR_GROUP + # FH
        [0.020]*BH_DR_GROUP) # BH


neighbour_weights_1stOrder = cms.vdouble(0, 0.25, 0,
                                         0.25, 0, 0.25,
                                         0, 0.25, 0)

neighbour_weights_2ndOrder = cms.vdouble(-0.25, 0.5, -0.25,
                                         0.5, 0,  0.5,
                                         -0.25, 0.5, -0.25)


seed_smoothing_ecal = cms.vdouble(
        1., 1., 1.,
        1., 1.1, 1.,
        1., 1., 1.,
        )
seed_smoothing_hcal = cms.vdouble(
        1., 1., 1., 1., 1.,
        1., 1., 1., 1., 1.,
        1., 1., 2., 1., 1.,
        1., 1., 1., 1., 1.,
        1., 1., 1., 1., 1.,
        )


distance_C3d_params = cms.PSet(type_multicluster=cms.string('dRC3d'),
                               dR_multicluster=cms.double(0.01),
                               minPt_multicluster=cms.double(0.5),  # minimum pt of the multicluster (GeV)
                               dist_dbscan_multicluster=cms.double(0.),
                               minN_dbscan_multicluster=cms.uint32(0),
                               EGIdentification=egamma_identification_drnn_cone.clone(),
                               )


dbscan_C3d_params = cms.PSet(type_multicluster=cms.string('DBSCANC3d'),
                             dR_multicluster=cms.double(0.),
                             minPt_multicluster=cms.double(0.5),  # minimum pt of the multicluster (GeV)
                             dist_dbscan_multicluster=cms.double(0.005),
                             minN_dbscan_multicluster=cms.uint32(3),
                             EGIdentification=egamma_identification_drnn_dbscan.clone())


histoMax_C3d_seeding_params = cms.PSet(type_histoalgo=cms.string('HistoMaxC3d'),
                               nBins_X1_histo_multicluster=cms.uint32(42), # bin size of about 0.012
                               nBins_X2_histo_multicluster=cms.uint32(216), # bin size of about 0.029
                               binSumsHisto=binSums,
                               kROverZMin=cms.double(0.076),
                               kROverZMax=cms.double(0.58),
                               threshold_histo_multicluster=cms.double(10.),
                               neighbour_weights=neighbour_weights_1stOrder,
                               seed_position=cms.string("TCWeighted"),#BinCentre, TCWeighted
                               seeding_space=cms.string("RPhi"),# RPhi, XY
                               seed_smoothing_ecal=seed_smoothing_ecal,
                               seed_smoothing_hcal=seed_smoothing_hcal,
                              )

## Note: this customization change both HGC and HFnose, to be revised
phase2_hfnose.toModify(histoMax_C3d_seeding_params,
                       nBins_X1_histo_multicluster=cms.uint32(43),
                       binSumsHisto=binSumsNose,
                       kROverZMin=cms.double(0.025),
                       kROverZMax=cms.double(0.58),
                       )

histoMax_C3d_clustering_params = cms.PSet(dR_multicluster=cms.double(0.03),
                               dR_multicluster_byLayer_coefficientA=cms.vdouble(),
                               dR_multicluster_byLayer_coefficientB=cms.vdouble(),
                               shape_threshold=cms.double(1.),
                               shape_distance=cms.double(0.015),
                               minPt_multicluster=cms.double(0.5),  # minimum pt of the multicluster (GeV)
                               cluster_association=cms.string("NearestNeighbour"),
                               EGIdentification=egamma_identification_histomax.clone(),
                               )


# V9 samples have a different defintion of the dEdx calibrations. To account for it
# we reascale the thresholds of the clustering seeds
# (see https://indico.cern.ch/event/806845/contributions/3359859/attachments/1815187/2966402/19-03-20_EGPerf_HGCBE.pdf
# for more details)
phase2_hgcalV9.toModify(histoMax_C3d_seeding_params,
                        threshold_histo_multicluster=7.5,  # MipT
                        )


histoMaxVariableDR_C3d_params = histoMax_C3d_clustering_params.clone(
        dR_multicluster = cms.double(0.),
        dR_multicluster_byLayer_coefficientA = cms.vdouble(dr_layerbylayer),
        dR_multicluster_byLayer_coefficientB = cms.vdouble([0]*(MAX_LAYERS+1))
        )


histoSecondaryMax_C3d_params = histoMax_C3d_seeding_params.clone(
        type_histoalgo = cms.string('HistoSecondaryMaxC3d')
        )

histoMaxXYVariableDR_C3d_params = histoMax_C3d_seeding_params.clone(
        seeding_space=cms.string("XY"),
        nBins_X1_histo_multicluster=cms.uint32(192),
        nBins_X2_histo_multicluster=cms.uint32(192)
        )

histoInterpolatedMax_C3d_params = histoMax_C3d_seeding_params.clone(
        type_histoalgo = cms.string('HistoInterpolatedMaxC3d')
        )


histoThreshold_C3d_params = histoMax_C3d_seeding_params.clone(
        type_histoalgo = cms.string('HistoThresholdC3d')
        )


histoMax_C3d_params = cms.PSet(
        type_multicluster=cms.string('Histo'),
        histoMax_C3d_clustering_parameters = histoMaxVariableDR_C3d_params.clone(),
        histoMax_C3d_seeding_parameters = histoMax_C3d_seeding_params.clone(),
        )


energy_interpretations_em = cms.PSet(type = cms.string('HGCalTriggerClusterInterpretationEM'),
                                     layer_containment_corrs = cms.vdouble(0., 0., 1.6144949, 0.92495334, 1.0820811, 0.9753549, 0.9742881, 1.0634482, 1.0599478, 0.9376349, 0.92587173, 0.8003076, 1.0417082, 1.7032381, 2.),
                                     scale_correction_coeff = cms.vdouble(16.68182373, -8.487143517),
                                     dr_bylayer = cms.vdouble([0.015]*15)
                                     )

phase2_hgcalV10.toModify(energy_interpretations_em,
                         layer_containment_corrs=cms.vdouble(0., 0., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.),
                         scale_correction_coeff=cms.vdouble(0., 0.),
                         )


energy_interpretations = cms.VPSet(energy_interpretations_em)

be_proc = cms.PSet(ProcessorName  = cms.string('HGCalBackendLayer2Processor3DClustering'),
                   C3d_parameters = histoMax_C3d_params.clone(),
                   energy_interpretations = energy_interpretations
                   )

hgcalBackEndLayer2Producer = cms.EDProducer(
    "HGCalBackendLayer2Producer",
    InputCluster = cms.InputTag('hgcalBackEndLayer1Producer:HGCalBackendLayer1Processor2DClustering'),
    ProcessorParameters = be_proc.clone()
    )
