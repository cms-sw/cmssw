import FWCore.ParameterSet.Config as cms

pfDeepBoostedJetPreprocessParams = cms.PSet(
    input_names = cms.vstring(
        'pfcand', 
        'sv'
    ),
    pfcand = cms.PSet(
        input_shape = cms.vuint32(1, 42, 100, 1),
        var_infos = cms.PSet(
            pfcand_VTX_ass = cms.PSet(
                median = cms.double(7.0),
                norm_factor = cms.double(1.0),
                replace_inf_value = cms.double(0),
                lower_bound = cms.double(-5),
                upper_bound = cms.double(5)
            ),
            pfcand_abseta = cms.PSet(
                median = cms.double(0.599505603313),
                norm_factor = cms.double(1.62486362457),
                replace_inf_value = cms.double(0),
                lower_bound = cms.double(-5),
                upper_bound = cms.double(5)
            ),
            pfcand_btagEtaRel = cms.PSet(
                median = cms.double(0.0),
                norm_factor = cms.double(0.309551209211),
                replace_inf_value = cms.double(0),
                lower_bound = cms.double(-5),
                upper_bound = cms.double(5)
            ),
            pfcand_btagJetDistVal = cms.PSet(
                median = cms.double(0.0),
                norm_factor = cms.double(1.0),
                replace_inf_value = cms.double(0),
                lower_bound = cms.double(-5),
                upper_bound = cms.double(5)
            ),
            pfcand_btagPParRatio = cms.PSet(
                median = cms.double(0.0),
                norm_factor = cms.double(1.00271141529),
                replace_inf_value = cms.double(0),
                lower_bound = cms.double(-5),
                upper_bound = cms.double(5)
            ),
            pfcand_btagPtRatio = cms.PSet(
                median = cms.double(0.0),
                norm_factor = cms.double(5.73484706879),
                replace_inf_value = cms.double(0),
                lower_bound = cms.double(-5),
                upper_bound = cms.double(5)
            ),
            pfcand_btagSip2dSig = cms.PSet(
                median = cms.double(0.0),
                norm_factor = cms.double(1.76905345917),
                replace_inf_value = cms.double(0),
                lower_bound = cms.double(-5),
                upper_bound = cms.double(5)
            ),
            pfcand_btagSip2dVal = cms.PSet(
                median = cms.double(0.0),
                norm_factor = cms.double(605.021118164),
                replace_inf_value = cms.double(0),
                lower_bound = cms.double(-5),
                upper_bound = cms.double(5)
            ),
            pfcand_btagSip3dSig = cms.PSet(
                median = cms.double(0.0),
                norm_factor = cms.double(0.966644644737),
                replace_inf_value = cms.double(0),
                lower_bound = cms.double(-5),
                upper_bound = cms.double(5)
            ),
            pfcand_btagSip3dVal = cms.PSet(
                median = cms.double(0.0),
                norm_factor = cms.double(239.708831787),
                replace_inf_value = cms.double(0),
                lower_bound = cms.double(-5),
                upper_bound = cms.double(5)
            ),
            pfcand_charge = cms.PSet(
                median = cms.double(0.0),
                norm_factor = cms.double(1.0),
                replace_inf_value = cms.double(0),
                lower_bound = cms.double(-5),
                upper_bound = cms.double(5)
            ),
            pfcand_deltaR = cms.PSet(
                median = cms.double(0.22575956583),
                norm_factor = cms.double(3.81050634384),
                replace_inf_value = cms.double(0),
                lower_bound = cms.double(-5),
                upper_bound = cms.double(5)
            ),
            pfcand_detadeta = cms.PSet(
                median = cms.double(0.0),
                norm_factor = cms.double(1967329.375),
                replace_inf_value = cms.double(0),
                lower_bound = cms.double(-5),
                upper_bound = cms.double(5)
            ),
            pfcand_dlambdadz = cms.PSet(
                median = cms.double(0.0),
                norm_factor = cms.double(1.0),
                replace_inf_value = cms.double(0),
                lower_bound = cms.double(-5),
                upper_bound = cms.double(5)
            ),
            pfcand_dphidphi = cms.PSet(
                median = cms.double(0.0),
                norm_factor = cms.double(1310150.125),
                replace_inf_value = cms.double(0),
                lower_bound = cms.double(-5),
                upper_bound = cms.double(5)
            ),
            pfcand_dphidxy = cms.PSet(
                median = cms.double(0.0),
                norm_factor = cms.double(1.0),
                replace_inf_value = cms.double(0),
                lower_bound = cms.double(-5),
                upper_bound = cms.double(5)
            ),
            pfcand_dptdpt = cms.PSet(
                median = cms.double(0.0),
                norm_factor = cms.double(163433.265625),
                replace_inf_value = cms.double(0),
                lower_bound = cms.double(-5),
                upper_bound = cms.double(5)
            ),
            pfcand_drminsv = cms.PSet(
                median = cms.double(0.177426457405),
                norm_factor = cms.double(2.64722108841),
                replace_inf_value = cms.double(-1),
                lower_bound = cms.double(-5),
                upper_bound = cms.double(5)
            ),
            pfcand_drsubjet1 = cms.PSet(
                median = cms.double(0.231124095619),
                norm_factor = cms.double(3.14072275162),
                replace_inf_value = cms.double(0),
                lower_bound = cms.double(-5),
                upper_bound = cms.double(5)
            ),
            pfcand_drsubjet2 = cms.PSet(
                median = cms.double(0.263272643089),
                norm_factor = cms.double(2.92228007317),
                replace_inf_value = cms.double(0),
                lower_bound = cms.double(-5),
                upper_bound = cms.double(5)
            ),
            pfcand_dxy = cms.PSet(
                median = cms.double(0.0),
                norm_factor = cms.double(392.713317871),
                replace_inf_value = cms.double(0),
                lower_bound = cms.double(-5),
                upper_bound = cms.double(5)
            ),
            pfcand_dxydxy = cms.PSet(
                median = cms.double(0.0),
                norm_factor = cms.double(46519.0234375),
                replace_inf_value = cms.double(0),
                lower_bound = cms.double(-5),
                upper_bound = cms.double(5)
            ),
            pfcand_dxydz = cms.PSet(
                median = cms.double(0.0),
                norm_factor = cms.double(7626007.0),
                replace_inf_value = cms.double(0),
                lower_bound = cms.double(-5),
                upper_bound = cms.double(5)
            ),
            pfcand_dxysig = cms.PSet(
                median = cms.double(0.0),
                norm_factor = cms.double(2.0138297081),
                replace_inf_value = cms.double(0),
                lower_bound = cms.double(-5),
                upper_bound = cms.double(5)
            ),
            pfcand_dz = cms.PSet(
                median = cms.double(0.0),
                norm_factor = cms.double(222.85093689),
                replace_inf_value = cms.double(0),
                lower_bound = cms.double(-5),
                upper_bound = cms.double(5)
            ),
            pfcand_dzdz = cms.PSet(
                median = cms.double(0.0),
                norm_factor = cms.double(29383.0683594),
                replace_inf_value = cms.double(0),
                lower_bound = cms.double(-5),
                upper_bound = cms.double(5)
            ),
            pfcand_dzsig = cms.PSet(
                median = cms.double(0.0),
                norm_factor = cms.double(1.67618751526),
                replace_inf_value = cms.double(0),
                lower_bound = cms.double(-5),
                upper_bound = cms.double(5)
            ),
            pfcand_erel_log = cms.PSet(
                median = cms.double(-5.38983869553),
                norm_factor = cms.double(0.539102613926),
                replace_inf_value = cms.double(-99),
                lower_bound = cms.double(-5),
                upper_bound = cms.double(5)
            ),
            pfcand_etarel = cms.PSet(
                median = cms.double(-0.0054658302106),
                norm_factor = cms.double(5.54556131363),
                replace_inf_value = cms.double(0),
                lower_bound = cms.double(-5),
                upper_bound = cms.double(5)
            ),
            pfcand_hcalFrac = cms.PSet(
                median = cms.double(0.0),
                norm_factor = cms.double(1.0),
                replace_inf_value = cms.double(0),
                lower_bound = cms.double(-5),
                upper_bound = cms.double(5)
            ),
            pfcand_isChargedHad = cms.PSet(
                median = cms.double(1.0),
                norm_factor = cms.double(1.0),
                replace_inf_value = cms.double(0),
                lower_bound = cms.double(-5),
                upper_bound = cms.double(5)
            ),
            pfcand_isEl = cms.PSet(
                median = cms.double(0.0),
                norm_factor = cms.double(1.0),
                replace_inf_value = cms.double(0),
                lower_bound = cms.double(-5),
                upper_bound = cms.double(5)
            ),
            pfcand_isGamma = cms.PSet(
                median = cms.double(0.0),
                norm_factor = cms.double(1.0),
                replace_inf_value = cms.double(0),
                lower_bound = cms.double(-5),
                upper_bound = cms.double(5)
            ),
            pfcand_isMu = cms.PSet(
                median = cms.double(0.0),
                norm_factor = cms.double(1.0),
                replace_inf_value = cms.double(0),
                lower_bound = cms.double(-5),
                upper_bound = cms.double(5)
            ),
            pfcand_isNeutralHad = cms.PSet(
                median = cms.double(0.0),
                norm_factor = cms.double(1.0),
                replace_inf_value = cms.double(0),
                lower_bound = cms.double(-5),
                upper_bound = cms.double(5)
            ),
            pfcand_lostInnerHits = cms.PSet(
                median = cms.double(-1.0),
                norm_factor = cms.double(1.0),
                replace_inf_value = cms.double(0),
                lower_bound = cms.double(-5),
                upper_bound = cms.double(5)
            ),
            pfcand_normchi2 = cms.PSet(
                median = cms.double(999.0),
                norm_factor = cms.double(1.0),
                replace_inf_value = cms.double(0),
                lower_bound = cms.double(-5),
                upper_bound = cms.double(5)
            ),
            pfcand_phirel = cms.PSet(
                median = cms.double(-5.10289683007e-05),
                norm_factor = cms.double(4.63705921173),
                replace_inf_value = cms.double(0),
                lower_bound = cms.double(-5),
                upper_bound = cms.double(5)
            ),
            pfcand_pt_log = cms.PSet(
                median = cms.double(1.09469842911),
                norm_factor = cms.double(0.518874108791),
                replace_inf_value = cms.double(-99),
                lower_bound = cms.double(-5),
                upper_bound = cms.double(5)
            ),
            pfcand_ptrel_log = cms.PSet(
                median = cms.double(-5.38205528259),
                norm_factor = cms.double(0.53791975975),
                replace_inf_value = cms.double(-99),
                lower_bound = cms.double(-5),
                upper_bound = cms.double(5)
            ),
            pfcand_puppiw = cms.PSet(
                median = cms.double(1.0),
                norm_factor = cms.double(1.0),
                replace_inf_value = cms.double(0),
                lower_bound = cms.double(-5),
                upper_bound = cms.double(5)
            ),
            pfcand_quality = cms.PSet(
                median = cms.double(0.0),
                norm_factor = cms.double(0.20000000298),
                replace_inf_value = cms.double(0),
                lower_bound = cms.double(-5),
                upper_bound = cms.double(5)
            )
        ),
        var_length = cms.uint32(100),
        var_names = cms.vstring(
            'pfcand_pt_log', 
            'pfcand_ptrel_log', 
            'pfcand_erel_log', 
            'pfcand_phirel', 
            'pfcand_etarel', 
            'pfcand_deltaR', 
            'pfcand_abseta', 
            'pfcand_puppiw', 
            'pfcand_drminsv', 
            'pfcand_drsubjet1', 
            'pfcand_drsubjet2', 
            'pfcand_charge', 
            'pfcand_isMu', 
            'pfcand_isEl', 
            'pfcand_isChargedHad', 
            'pfcand_isGamma', 
            'pfcand_isNeutralHad', 
            'pfcand_hcalFrac', 
            'pfcand_VTX_ass', 
            'pfcand_lostInnerHits', 
            'pfcand_normchi2', 
            'pfcand_quality', 
            'pfcand_dz', 
            'pfcand_dzsig', 
            'pfcand_dxy', 
            'pfcand_dxysig', 
            'pfcand_dptdpt', 
            'pfcand_detadeta', 
            'pfcand_dphidphi', 
            'pfcand_dxydxy', 
            'pfcand_dzdz', 
            'pfcand_dxydz', 
            'pfcand_dphidxy', 
            'pfcand_dlambdadz', 
            'pfcand_btagEtaRel', 
            'pfcand_btagPtRatio', 
            'pfcand_btagPParRatio', 
            'pfcand_btagSip2dVal', 
            'pfcand_btagSip2dSig', 
            'pfcand_btagSip3dVal', 
            'pfcand_btagSip3dSig', 
            'pfcand_btagJetDistVal'
        )
    ),
    sv = cms.PSet(
        input_shape = cms.vuint32(1, 15, 7, 1),
        var_infos = cms.PSet(
            sv_abseta = cms.PSet(
                median = cms.double(0.579698801041),
                norm_factor = cms.double(1.54787838459),
                replace_inf_value = cms.double(0),
                lower_bound = cms.double(-5),
                upper_bound = cms.double(5)
            ),
            sv_costhetasvpv = cms.PSet(
                median = cms.double(0.999744534492),
                norm_factor = cms.double(4035.89501953),
                replace_inf_value = cms.double(0),
                lower_bound = cms.double(-5),
                upper_bound = cms.double(5)
            ),
            sv_d3d = cms.PSet(
                median = cms.double(0.496477723122),
                norm_factor = cms.double(0.230671063066),
                replace_inf_value = cms.double(0),
                lower_bound = cms.double(-5),
                upper_bound = cms.double(5)
            ),
            sv_d3dsig = cms.PSet(
                median = cms.double(6.50515079498),
                norm_factor = cms.double(0.0328928083181),
                replace_inf_value = cms.double(0),
                lower_bound = cms.double(-5),
                upper_bound = cms.double(5)
            ),
            sv_deltaR = cms.PSet(
                median = cms.double(0.105607174337),
                norm_factor = cms.double(4.58211994171),
                replace_inf_value = cms.double(0),
                lower_bound = cms.double(-5),
                upper_bound = cms.double(5)
            ),
            sv_dxy = cms.PSet(
                median = cms.double(0.369336694479),
                norm_factor = cms.double(0.277160406113),
                replace_inf_value = cms.double(0),
                lower_bound = cms.double(-5),
                upper_bound = cms.double(5)
            ),
            sv_dxysig = cms.PSet(
                median = cms.double(6.49603271484),
                norm_factor = cms.double(0.0328965261579),
                replace_inf_value = cms.double(0),
                lower_bound = cms.double(-5),
                upper_bound = cms.double(5)
            ),
            sv_erel_log = cms.PSet(
                median = cms.double(-2.77275466919),
                norm_factor = cms.double(0.976448118687),
                replace_inf_value = cms.double(-99),
                lower_bound = cms.double(-5),
                upper_bound = cms.double(5)
            ),
            sv_etarel = cms.PSet(
                median = cms.double(-0.00318800867535),
                norm_factor = cms.double(10.1732673645),
                replace_inf_value = cms.double(0),
                lower_bound = cms.double(-5),
                upper_bound = cms.double(5)
            ),
            sv_mass = cms.PSet(
                median = cms.double(1.36728298664),
                norm_factor = cms.double(0.294783204794),
                replace_inf_value = cms.double(0),
                lower_bound = cms.double(-5),
                upper_bound = cms.double(5)
            ),
            sv_normchi2 = cms.PSet(
                median = cms.double(0.796081960201),
                norm_factor = cms.double(0.737576127052),
                replace_inf_value = cms.double(0),
                lower_bound = cms.double(-5),
                upper_bound = cms.double(5)
            ),
            sv_ntracks = cms.PSet(
                median = cms.double(3.0),
                norm_factor = cms.double(0.5),
                replace_inf_value = cms.double(0),
                lower_bound = cms.double(-5),
                upper_bound = cms.double(5)
            ),
            sv_phirel = cms.PSet(
                median = cms.double(0.000510219600983),
                norm_factor = cms.double(9.11781406403),
                replace_inf_value = cms.double(0),
                lower_bound = cms.double(-5),
                upper_bound = cms.double(5)
            ),
            sv_pt_log = cms.PSet(
                median = cms.double(3.77797603607),
                norm_factor = cms.double(0.889925837517),
                replace_inf_value = cms.double(-99),
                lower_bound = cms.double(-5),
                upper_bound = cms.double(5)
            ),
            sv_ptrel_log = cms.PSet(
                median = cms.double(-2.7622461319),
                norm_factor = cms.double(0.975634276867),
                replace_inf_value = cms.double(-99),
                lower_bound = cms.double(-5),
                upper_bound = cms.double(5)
            )
        ),
        var_length = cms.uint32(7),
        var_names = cms.vstring(
            'sv_pt_log', 
            'sv_ptrel_log', 
            'sv_erel_log', 
            'sv_phirel', 
            'sv_etarel', 
            'sv_deltaR', 
            'sv_abseta', 
            'sv_mass', 
            'sv_ntracks', 
            'sv_normchi2', 
            'sv_dxy', 
            'sv_dxysig', 
            'sv_d3d', 
            'sv_d3dsig', 
            'sv_costhetasvpv'
        )
    )
)
