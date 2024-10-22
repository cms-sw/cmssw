import FWCore.ParameterSet.Config as cms

from Configuration.Eras.Modifier_phase2_hgcalV10_cff import phase2_hgcalV10
from Configuration.Eras.Modifier_phase2_hgcalV11_cff import phase2_hgcalV11

inputs_small = ['cl3d_firstlayer', 'cl3d_coreshowerlength', 'cl3d_maxlayer', 'cl3d_srrmean']
inputs_large = ['cl3d_coreshowerlength', 'cl3d_showerlength', 'cl3d_firstlayer', 'cl3d_maxlayer', 'cl3d_szz', 'cl3d_srrmean', 'cl3d_srrtot', 'cl3d_seetot', 'cl3d_spptot']

class Category:
    def __init__(self, eta_min, eta_max, pt_min, pt_max):
        self.eta_min = eta_min
        self.eta_max = eta_max
        self.pt_min = pt_min
        self.pt_max = pt_max


categories = [
        # Low eta
        Category(eta_min=1.5, eta_max=2.7, pt_min=0., pt_max=1e6),
        # High eta
        Category(eta_min=2.7, eta_max=3.0, pt_min=0., pt_max=1e6),
        ]

# Identification for dRNN 2D clustering and cone 3D clustering
bdt_weights_drnn_cone = [
        # Low eta
        'L1Trigger/L1THGCal/data/egamma_id_drnn_cone_loweta_v0.xml',
        # High eta
        'L1Trigger/L1THGCal/data/egamma_id_drnn_cone_higheta_v0.xml',
        ]

working_points_drnn_cone = [
        # Low eta
        {
        '900':0.14057436,
        '950':0.05661769,
        '975':-0.01481255,
        '995':-0.19656579,
        },
        # High eta
        {
        '900':0.05995301,
        '950':-0.02947988,
        '975':-0.10577436,
        '995':-0.26401181,
        },
        ]


# Identification for dRNN 2D clustering and DBSCAN 3D clustering
bdt_weights_drnn_dbscan = [
        # Low eta
        'L1Trigger/L1THGCal/data/egamma_id_drnn_dbscan_loweta_v0.xml',
        # High eta
        'L1Trigger/L1THGCal/data/egamma_id_drnn_dbscan_higheta_v0.xml',
        ]

working_points_drnn_dbscan = [
        # Low eta
        {
        '900':0.08421164,
        '950':0.06436077,
        '975':-0.04547527,
        '995':-0.23717142,
        },
        # High eta
        {
        '900':0.05559443,
        '950':-0.0171725,
        '975':-0.10630798,
        '995':-0.27290947,
        },
        ]

# Identification for 3D HistoMax clustering: switch between configurations for v8 and v9 geometry
input_features_histomax = {
        "v8_352":inputs_small,
        "v9_370":inputs_large,
        "v9_394":inputs_large,
        "v10_3151":inputs_large
        }

bdt_weights_histomax = {
        "v8_352":[ #trained using TPG software version 3.5.2
                # Low eta
                'L1Trigger/L1THGCal/data/egamma_id_histomax_352_loweta_v0.xml',
                # High eta
                'L1Trigger/L1THGCal/data/egamma_id_histomax_352_higheta_v0.xml'
                ],
        "v9_370":[ #trained using TPG software version 3.7.0
                # Low eta
                'L1Trigger/L1THGCal/data/egamma_id_histomax_370_loweta_v0.xml',
                # High eta
                'L1Trigger/L1THGCal/data/egamma_id_histomax_370_higheta_v0.xml'
                ],
        "v9_394":[ #trained using TPG software version 3.9.4
                # Low eta
                'L1Trigger/L1THGCal/data/egamma_id_histomax_394_loweta_v0.xml',
                # High eta
                'L1Trigger/L1THGCal/data/egamma_id_histomax_394_higheta_v0.xml'
                ],
        "v10_3151":[ #trained using TPG software version 3.15.1
                # Low eta
                'L1Trigger/L1THGCal/data/egamma_id_histomax_3151_loweta_v0.xml',
                # High eta
                'L1Trigger/L1THGCal/data/egamma_id_histomax_3151_higheta_v0.xml'
                ]
        }

working_points_histomax = {
        "v8_352":[
                # Low eta
                {
                '900':0.19146989,
                '950':0.1379665,
                '975':0.03496629,
                '995':-0.24383164,
                },
                # High eta
                {
                '900':0.13347613,
                '950':0.04267797,
                '975':-0.03698097,
                '995':-0.23077505,
                }
             ],
        "v9_370":[
                # Low eta
                {
                '900':0.8815851, # epsilon_b = 2.5%
                '950':0.5587649, # epsilon_b = 4.0%
                '975':-0.1937952, # epsilon_b = 5.5%
                '995':-0.9394884, # epsilon_b = 10.1%
                },
                # High eta
                {
                '900':0.7078400, #epsilon_b = 3.5%
                '950':-0.0239623, #epsilon_b = 6.9%
                '975':-0.7045071, #epsilon_b = 11.6%
                '995':-0.9811426, #epsilon_b = 26.1%
                }
             ],
        "v9_394":[
                # Low eta
                {
                '900':0.9794103, # epsilon_b = 1.9%
                '950':0.9052764, # epsilon_b = 3.9%
                '975':0.5276631, # epsilon_b = 6.6%
                '995':-0.9153535, # epsilon_b = 20.4%
                },
                # High eta
                {
                '900':0.8825340, #epsilon_b = 1.0%
                '950':0.2856039, #epsilon_b = 1.7%
                '975':-0.5274948, #epsilon_b = 2.8%
                '995':-0.9864445, #epsilon_b = 7.6%
                }
             ],
        "v10_3151": [
                # Low eta
                {
                 '900': 0.9903189,
                 '950': 0.9646683,
                 '975': 0.8292287,
                 '995': -0.7099538,
                },
                # High eta
                {
                 '900': 0.9932326,
                 '950': 0.9611762,
                 '975': 0.7616282,
                 '995': -0.9163715,
                }
             ]
        }

tight_wp = ['975', '900']
loose_wp = ['995', '950']


egamma_identification_drnn_cone = cms.PSet(
        Inputs=cms.vstring(inputs_small),
        CategoriesEtaMin=cms.vdouble([cat.eta_min for cat in categories]),
        CategoriesEtaMax=cms.vdouble([cat.eta_max for cat in categories]),
        CategoriesPtMin=cms.vdouble([cat.pt_min for cat in categories]),
        CategoriesPtMax=cms.vdouble([cat.pt_max for cat in categories]),
        Weights=cms.vstring(bdt_weights_drnn_cone),
        WorkingPoints=cms.vdouble([wps[eff] for wps,eff in zip(working_points_drnn_cone,tight_wp)]),
        )

egamma_identification_drnn_dbscan = cms.PSet(
        Inputs=cms.vstring(inputs_small),
        CategoriesEtaMin=cms.vdouble([cat.eta_min for cat in categories]),
        CategoriesEtaMax=cms.vdouble([cat.eta_max for cat in categories]),
        CategoriesPtMin=cms.vdouble([cat.pt_min for cat in categories]),
        CategoriesPtMax=cms.vdouble([cat.pt_max for cat in categories]),
        Weights=cms.vstring(bdt_weights_drnn_dbscan),
        WorkingPoints=cms.vdouble([wps[eff] for wps,eff in zip(working_points_drnn_dbscan,tight_wp)]),
        )

egamma_identification_histomax = cms.PSet(
        Inputs=cms.vstring(input_features_histomax['v10_3151']),
        CategoriesEtaMin=cms.vdouble([cat.eta_min for cat in categories]),
        CategoriesEtaMax=cms.vdouble([cat.eta_max for cat in categories]),
        CategoriesPtMin=cms.vdouble([cat.pt_min for cat in categories]),
        CategoriesPtMax=cms.vdouble([cat.pt_max for cat in categories]),
        Weights=cms.vstring(bdt_weights_histomax['v10_3151']),
        WorkingPoints=cms.VPSet([
            cms.PSet(
                Name=cms.string('tight'),
                WorkingPoint=cms.vdouble([wps[eff] for wps,eff in zip(working_points_histomax['v10_3151'],tight_wp)])
            ),
            cms.PSet(
                Name=cms.string('loose'),
                WorkingPoint=cms.vdouble([wps[eff] for wps,eff in zip(working_points_histomax['v10_3151'],loose_wp)])
            ),
            ])
        )

phase2_hgcalV10.toModify(egamma_identification_histomax,
        Inputs=cms.vstring(input_features_histomax['v10_3151']),
        Weights=cms.vstring(bdt_weights_histomax['v10_3151']),
        WorkingPoints=cms.VPSet([
            cms.PSet(
                Name=cms.string('tight'),
                WorkingPoint=cms.vdouble([wps[eff] for wps,eff in zip(working_points_histomax['v10_3151'],tight_wp)])
            ),
            cms.PSet(
                Name=cms.string('loose'),
                WorkingPoint=cms.vdouble([wps[eff] for wps,eff in zip(working_points_histomax['v10_3151'],loose_wp)])
            ),
            ])
        )

phase2_hgcalV11.toModify(egamma_identification_histomax,
        Inputs=cms.vstring(input_features_histomax['v10_3151']),
        Weights=cms.vstring(bdt_weights_histomax['v10_3151']),
        WorkingPoints=cms.VPSet([
            cms.PSet(
                Name=cms.string('tight'),
                WorkingPoint=cms.vdouble([wps[eff] for wps,eff in zip(working_points_histomax['v10_3151'],tight_wp)])
            ),
            cms.PSet(
                Name=cms.string('loose'),
                WorkingPoint=cms.vdouble([wps[eff] for wps,eff in zip(working_points_histomax['v10_3151'],loose_wp)])
            ),
            ])
        )
