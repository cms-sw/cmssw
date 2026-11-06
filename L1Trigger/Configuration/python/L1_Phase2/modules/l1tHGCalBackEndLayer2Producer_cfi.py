import FWCore.ParameterSet.Config as cms

l1tHGCalBackEndLayer2Producer = cms.EDProducer("HGCalBackendLayer2Producer",
    InputCluster = cms.InputTag("l1tHGCalBackEndLayer1Producer","HGCalBackendLayer1Processor2DClustering"),
    ProcessorParameters = cms.PSet(
        C3d_parameters = cms.PSet(
            histoMax_C3d_clustering_parameters = cms.PSet(
                AlgoName = cms.string('HGCalHistoClusteringWrapper'),
                EGIdentification = cms.PSet(
                    CategoriesEtaMax = cms.vdouble(2.7, 3.0),
                    CategoriesEtaMin = cms.vdouble(1.5, 2.7),
                    CategoriesPtMax = cms.vdouble(1000000.0, 1000000.0),
                    CategoriesPtMin = cms.vdouble(0.0, 0.0),
                    Inputs = cms.vstring(
                        'cl3d_coreshowerlength',
                        'cl3d_showerlength',
                        'cl3d_firstlayer',
                        'cl3d_maxlayer',
                        'cl3d_szz',
                        'cl3d_srrmean',
                        'cl3d_srrtot',
                        'cl3d_seetot',
                        'cl3d_spptot'
                    ),
                    Weights = cms.vstring(
                        'L1Trigger/L1THGCal/data/egamma_id_histomax_3151_loweta_v0.xml',
                        'L1Trigger/L1THGCal/data/egamma_id_histomax_3151_higheta_v0.xml'
                    ),
                    WorkingPoints = cms.VPSet(
                        cms.PSet(
                            Name = cms.string('tight'),
                            WorkingPoint = cms.vdouble(0.8292287, 0.9932326)
                        ),
                        cms.PSet(
                            Name = cms.string('loose'),
                            WorkingPoint = cms.vdouble(-0.7099538, 0.9611762)
                        )
                    )
                ),
                cluster_association = cms.string('NearestNeighbour'),
                dR_multicluster = cms.double(0.0),
                dR_multicluster_byLayer_coefficientA = cms.vdouble(
                    0, 0.015, 0.015, 0.015, 0.015,
                    0.015, 0.015, 0.015, 0.02, 0.02,
                    0.02, 0.02, 0.02, 0.02, 0.02,
                    0.03, 0.03, 0.03, 0.03, 0.03,
                    0.03, 0.03, 0.04, 0.04, 0.04,
                    0.04, 0.04, 0.04, 0.04, 0.04,
                    0.04, 0.04, 0.04, 0.04, 0.04,
                    0.05, 0.05, 0.05, 0.05, 0.05,
                    0.05, 0.05, 0.05, 0.05, 0.05,
                    0.05, 0.05, 0.05, 0.05, 0.05,
                    0.05, 0.05, 0.05
                ),
                dR_multicluster_byLayer_coefficientB = cms.vdouble(
                    0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0,
                    0, 0, 0
                ),
                minPt_multicluster = cms.double(0.5),
                shape_distance = cms.double(0.015),
                shape_threshold = cms.double(1.0)
            ),
            histoMax_C3d_seeding_parameters = cms.PSet(
                binSumsHisto = cms.vuint32(
                    13, 11, 11, 11, 9,
                    9, 9, 7, 7, 7,
                    7, 7, 7, 5, 5,
                    5, 5, 5, 5, 5,
                    5, 5, 5, 5, 5,
                    5, 5, 5, 3, 3,
                    3, 3, 3, 3, 3,
                    3, 3, 3, 3, 3,
                    3, 3
                ),
                kROverZMax = cms.double(0.58),
                kROverZMin = cms.double(0.076),
                nBins_X1_histo_multicluster = cms.uint32(42),
                nBins_X2_histo_multicluster = cms.uint32(216),
                neighbour_weights = cms.vdouble(
                    0, 0.25, 0, 0.25, 0,
                    0.25, 0, 0.25, 0
                ),
                seed_position = cms.string('TCWeighted'),
                seed_smoothing_ecal = cms.vdouble(
                    1.0, 1.0, 1.0, 1.0, 1.1,
                    1.0, 1.0, 1.0, 1.0
                ),
                seed_smoothing_hcal = cms.vdouble(
                    1.0, 1.0, 1.0, 1.0, 1.0,
                    1.0, 1.0, 1.0, 1.0, 1.0,
                    1.0, 1.0, 2.0, 1.0, 1.0,
                    1.0, 1.0, 1.0, 1.0, 1.0,
                    1.0, 1.0, 1.0, 1.0, 1.0
                ),
                seeding_space = cms.string('RPhi'),
                seeds_norm_by_area = cms.bool(False),
                threshold_histo_multicluster = cms.double(20.0),
                type_histoalgo = cms.string('HistoMaxC3d')
            ),
            histoMax_C3d_sorting_truncation_parameters = cms.PSet(
                AlgoName = cms.string('HGCalSortingTruncationWrapper'),
                maxTCs = cms.uint32(80)
            ),
            type_multicluster = cms.string('Histo')
        ),
        ProcessorName = cms.string('HGCalBackendLayer2Processor3DClustering'),
        energy_interpretations = cms.VPSet(cms.PSet(
            dr_bylayer = cms.vdouble(
                0.015, 0.015, 0.015, 0.015, 0.015,
                0.015, 0.015, 0.015, 0.015, 0.015,
                0.015, 0.015, 0.015, 0.015, 0.015
            ),
            layer_containment_corrs = cms.vdouble(
                0.0, 0.0, 1.28, 1.09, 1.0,
                1.07, 1.09, 1.04, 1.0, 1.09,
                1.07, 1.03, 0.93, 1.4, 1.89
            ),
            scale_correction_coeff = cms.vdouble(52.99, -24.96),
            type = cms.string('HGCalTriggerClusterInterpretationEM')
        ))
    )
)
