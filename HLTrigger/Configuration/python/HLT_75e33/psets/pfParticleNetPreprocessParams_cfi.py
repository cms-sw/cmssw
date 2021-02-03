import FWCore.ParameterSet.Config as cms

pfParticleNetPreprocessParams = cms.PSet(
    input_names = cms.vstring(
        'pf_points',
        'pf_features',
        'pf_mask',
        'sv_points',
        'sv_features',
        'sv_mask'
    ),
    pf_features = cms.PSet(
        input_shape = cms.vuint32(1, 25, 100),
        var_infos = cms.PSet(
            pfcand_VTX_ass = cms.PSet(
                median = cms.double(7.0),
                norm_factor = cms.double(1.0)
            ),
            pfcand_abseta = cms.PSet(
                median = cms.double(0.570207834244),
                norm_factor = cms.double(1.65039794176)
            ),
            pfcand_btagEtaRel = cms.PSet(
                median = cms.double(1.02164459229),
                norm_factor = cms.double(0.450494647657)
            ),
            pfcand_btagJetDistVal = cms.PSet(
                median = cms.double(-8.01788191893e-05),
                norm_factor = cms.double(172.900234853)
            ),
            pfcand_btagPParRatio = cms.PSet(
                median = cms.double(0.777632951736),
                norm_factor = cms.double(1.28595373661)
            ),
            pfcand_btagPtRatio = cms.PSet(
                median = cms.double(0.00793754169717),
                norm_factor = cms.double(4.41778965372)
            ),
            pfcand_btagSip3dSig = cms.PSet(
                median = cms.double(0.0),
                norm_factor = cms.double(0.871534329165)
            ),
            pfcand_btagSip3dVal = cms.PSet(
                median = cms.double(0.0),
                norm_factor = cms.double(252.165554148)
            ),
            pfcand_charge = cms.PSet(
                median = cms.double(0.0),
                norm_factor = cms.double(1.0)
            ),
            pfcand_dxy = cms.PSet(
                median = cms.double(0.0),
                norm_factor = cms.double(473.799859238)
            ),
            pfcand_dxysig = cms.PSet(
                median = cms.double(0.0),
                norm_factor = cms.double(1.66318902024)
            ),
            pfcand_dz = cms.PSet(
                median = cms.double(0.0),
                norm_factor = cms.double(297.458232668)
            ),
            pfcand_dzsig = cms.PSet(
                median = cms.double(0.0),
                norm_factor = cms.double(1.38690098449)
            ),
            pfcand_e_log_nopuppi = cms.PSet(
                median = cms.double(1.42155307531),
                norm_factor = cms.double(0.519927136489)
            ),
            pfcand_etarel = cms.PSet(
                median = cms.double(-0.00481873401441),
                norm_factor = cms.double(4.44798275517)
            ),
            pfcand_isChargedHad = cms.PSet(
                median = cms.double(1.0),
                norm_factor = cms.double(1.0)
            ),
            pfcand_isEl = cms.PSet(
                median = cms.double(0.0),
                norm_factor = cms.double(1.0)
            ),
            pfcand_isGamma = cms.PSet(
                median = cms.double(0.0),
                norm_factor = cms.double(1.0)
            ),
            pfcand_isMu = cms.PSet(
                median = cms.double(0.0),
                norm_factor = cms.double(1.0)
            ),
            pfcand_isNeutralHad = cms.PSet(
                median = cms.double(0.0),
                norm_factor = cms.double(1.0)
            ),
            pfcand_lostInnerHits = cms.PSet(
                median = cms.double(-1.0),
                norm_factor = cms.double(1.0)
            ),
            pfcand_normchi2 = cms.PSet(
                median = cms.double(3.0),
                norm_factor = cms.double(0.00100401606426)
            ),
            pfcand_phirel = cms.PSet(
                median = cms.double(-0.000108399453893),
                norm_factor = cms.double(4.39792304438)
            ),
            pfcand_pt_log_nopuppi = cms.PSet(
                median = cms.double(1.12241530418),
                norm_factor = cms.double(0.514354843556)
            ),
            pfcand_quality = cms.PSet(
                median = cms.double(5.0),
                norm_factor = cms.double(0.2)
            )
        ),
        var_length = cms.uint32(100),
        var_names = cms.vstring(
            'pfcand_pt_log_nopuppi',
            'pfcand_e_log_nopuppi',
            'pfcand_etarel',
            'pfcand_phirel',
            'pfcand_abseta',
            'pfcand_charge',
            'pfcand_isMu',
            'pfcand_isEl',
            'pfcand_isChargedHad',
            'pfcand_isGamma',
            'pfcand_isNeutralHad',
            'pfcand_VTX_ass',
            'pfcand_lostInnerHits',
            'pfcand_normchi2',
            'pfcand_quality',
            'pfcand_dz',
            'pfcand_dzsig',
            'pfcand_dxy',
            'pfcand_dxysig',
            'pfcand_btagEtaRel',
            'pfcand_btagPtRatio',
            'pfcand_btagPParRatio',
            'pfcand_btagSip3dVal',
            'pfcand_btagSip3dSig',
            'pfcand_btagJetDistVal'
        )
    ),
    pf_mask = cms.PSet(
        input_shape = cms.vuint32(1, 1, 100),
        var_infos = cms.PSet(
            pfcand_mask = cms.PSet(
                median = cms.double(0.0),
                norm_factor = cms.double(1.0)
            )
        ),
        var_length = cms.uint32(100),
        var_names = cms.vstring('pfcand_mask')
    ),
    pf_points = cms.PSet(
        input_shape = cms.vuint32(1, 2, 100),
        var_infos = cms.PSet(
            pfcand_etarel = cms.PSet(
                median = cms.double(-0.00481873401441),
                norm_factor = cms.double(4.44798275517)
            ),
            pfcand_phirel = cms.PSet(
                median = cms.double(-0.000108399453893),
                norm_factor = cms.double(4.39792304438)
            )
        ),
        var_length = cms.uint32(100),
        var_names = cms.vstring(
            'pfcand_etarel',
            'pfcand_phirel'
        )
    ),
    sv_features = cms.PSet(
        input_shape = cms.vuint32(1, 12, 7),
        var_infos = cms.PSet(
            sv_abseta = cms.PSet(
                median = cms.double(0.549196600914),
                norm_factor = cms.double(1.59394768801)
            ),
            sv_costhetasvpv = cms.PSet(
                median = cms.double(0.999762713909),
                norm_factor = cms.double(183.233393766)
            ),
            sv_d3d = cms.PSet(
                median = cms.double(0.551342129707),
                norm_factor = cms.double(0.223128153806)
            ),
            sv_d3dsig = cms.PSet(
                median = cms.double(8.35655975342),
                norm_factor = cms.double(0.0238999954695)
            ),
            sv_dxy = cms.PSet(
                median = cms.double(0.421502441168),
                norm_factor = cms.double(0.312218277088)
            ),
            sv_dxysig = cms.PSet(
                median = cms.double(8.34426879883),
                norm_factor = cms.double(0.0239017055779)
            ),
            sv_etarel = cms.PSet(
                median = cms.double(-0.00247404468246),
                norm_factor = cms.double(8.44460188362)
            ),
            sv_mass = cms.PSet(
                median = cms.double(1.3346953392),
                norm_factor = cms.double(0.309230848837)
            ),
            sv_normchi2 = cms.PSet(
                median = cms.double(0.806915462017),
                norm_factor = cms.double(0.724496098658)
            ),
            sv_ntracks = cms.PSet(
                median = cms.double(3.0),
                norm_factor = cms.double(0.5)
            ),
            sv_phirel = cms.PSet(
                median = cms.double(0.000502154347487),
                norm_factor = cms.double(8.73243388771)
            ),
            sv_pt_log = cms.PSet(
                median = cms.double(3.78325366974),
                norm_factor = cms.double(0.692538144148)
            )
        ),
        var_length = cms.uint32(7),
        var_names = cms.vstring(
            'sv_pt_log',
            'sv_mass',
            'sv_phirel',
            'sv_etarel',
            'sv_abseta',
            'sv_ntracks',
            'sv_normchi2',
            'sv_dxy',
            'sv_dxysig',
            'sv_d3d',
            'sv_d3dsig',
            'sv_costhetasvpv'
        )
    ),
    sv_mask = cms.PSet(
        input_shape = cms.vuint32(1, 1, 7),
        var_infos = cms.PSet(
            sv_mask = cms.PSet(
                median = cms.double(0.0),
                norm_factor = cms.double(1.0)
            )
        ),
        var_length = cms.uint32(7),
        var_names = cms.vstring('sv_mask')
    ),
    sv_points = cms.PSet(
        input_shape = cms.vuint32(1, 2, 7),
        var_infos = cms.PSet(
            sv_etarel = cms.PSet(
                median = cms.double(-0.00247404468246),
                norm_factor = cms.double(8.44460188362)
            ),
            sv_phirel = cms.PSet(
                median = cms.double(0.000502154347487),
                norm_factor = cms.double(8.73243388771)
            )
        ),
        var_length = cms.uint32(7),
        var_names = cms.vstring(
            'sv_phirel',
            'sv_etarel'
        )
    )
)