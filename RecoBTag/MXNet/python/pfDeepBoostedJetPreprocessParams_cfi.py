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
                upper = cms.double(7.0)
            ),
            pfcand_abseta = cms.PSet(
                median = cms.double(0.599505603313),
                upper = cms.double(1.21494185925)
            ),
            pfcand_btagEtaRel = cms.PSet(
                median = cms.double(0.0),
                upper = cms.double(3.23048327446)
            ),
            pfcand_btagJetDistVal = cms.PSet(
                median = cms.double(0.0),
                upper = cms.double(0.0)
            ),
            pfcand_btagPParRatio = cms.PSet(
                median = cms.double(0.0),
                upper = cms.double(0.997295975685)
            ),
            pfcand_btagPtRatio = cms.PSet(
                median = cms.double(0.0),
                upper = cms.double(0.174372568727)
            ),
            pfcand_btagSip2dSig = cms.PSet(
                median = cms.double(0.0),
                upper = cms.double(0.565274050236)
            ),
            pfcand_btagSip2dVal = cms.PSet(
                median = cms.double(0.0),
                upper = cms.double(0.00165283482056)
            ),
            pfcand_btagSip3dSig = cms.PSet(
                median = cms.double(0.0),
                upper = cms.double(1.03450630188)
            ),
            pfcand_btagSip3dVal = cms.PSet(
                median = cms.double(0.0),
                upper = cms.double(0.00417172765359)
            ),
            pfcand_charge = cms.PSet(
                median = cms.double(0.0),
                upper = cms.double(1.0)
            ),
            pfcand_deltaR = cms.PSet(
                median = cms.double(0.22575956583),
                upper = cms.double(0.488191870451)
            ),
            pfcand_detadeta = cms.PSet(
                median = cms.double(0.0),
                upper = cms.double(5.08303287461e-07)
            ),
            pfcand_dlambdadz = cms.PSet(
                median = cms.double(0.0),
                upper = cms.double(0.0)
            ),
            pfcand_dphidphi = cms.PSet(
                median = cms.double(0.0),
                upper = cms.double(7.63271316373e-07)
            ),
            pfcand_dphidxy = cms.PSet(
                median = cms.double(0.0),
                upper = cms.double(0.0)
            ),
            pfcand_dptdpt = cms.PSet(
                median = cms.double(0.0),
                upper = cms.double(6.11870564171e-06)
            ),
            pfcand_drminsv = cms.PSet(
                median = cms.double(0.177426457405),
                upper = cms.double(0.555181086063)
            ),
            pfcand_drsubjet1 = cms.PSet(
                median = cms.double(0.231124095619),
                upper = cms.double(0.549522156715)
            ),
            pfcand_drsubjet2 = cms.PSet(
                median = cms.double(0.263272643089),
                upper = cms.double(0.605471189022)
            ),
            pfcand_dxy = cms.PSet(
                median = cms.double(0.0),
                upper = cms.double(0.00254638679326)
            ),
            pfcand_dxydxy = cms.PSet(
                median = cms.double(0.0),
                upper = cms.double(2.14965821215e-05)
            ),
            pfcand_dxydz = cms.PSet(
                median = cms.double(0.0),
                upper = cms.double(1.31130221348e-07)
            ),
            pfcand_dxysig = cms.PSet(
                median = cms.double(0.0),
                upper = cms.double(0.496566288471)
            ),
            pfcand_dz = cms.PSet(
                median = cms.double(0.0),
                upper = cms.double(0.00448730448261)
            ),
            pfcand_dzdz = cms.PSet(
                median = cms.double(0.0),
                upper = cms.double(3.40332044289e-05)
            ),
            pfcand_dzsig = cms.PSet(
                median = cms.double(0.0),
                upper = cms.double(0.596591930389)
            ),
            pfcand_erel_log = cms.PSet(
                median = cms.double(-5.38983869553),
                upper = cms.double(-3.53490426064)
            ),
            pfcand_etarel = cms.PSet(
                median = cms.double(-0.0054658302106),
                upper = cms.double(0.174858552814)
            ),
            pfcand_hcalFrac = cms.PSet(
                median = cms.double(0.0),
                upper = cms.double(0.0)
            ),
            pfcand_isChargedHad = cms.PSet(
                median = cms.double(1.0),
                upper = cms.double(1.0)
            ),
            pfcand_isEl = cms.PSet(
                median = cms.double(0.0),
                upper = cms.double(0.0)
            ),
            pfcand_isGamma = cms.PSet(
                median = cms.double(0.0),
                upper = cms.double(1.0)
            ),
            pfcand_isMu = cms.PSet(
                median = cms.double(0.0),
                upper = cms.double(0.0)
            ),
            pfcand_isNeutralHad = cms.PSet(
                median = cms.double(0.0),
                upper = cms.double(0.0)
            ),
            pfcand_lostInnerHits = cms.PSet(
                median = cms.double(-1.0),
                upper = cms.double(-1.0)
            ),
            pfcand_normchi2 = cms.PSet(
                median = cms.double(999.0),
                upper = cms.double(999.0)
            ),
            pfcand_phirel = cms.PSet(
                median = cms.double(-5.10289683007e-05),
                upper = cms.double(0.215602903366)
            ),
            pfcand_pt_log = cms.PSet(
                median = cms.double(1.09469842911),
                upper = cms.double(3.02194809914)
            ),
            pfcand_ptrel_log = cms.PSet(
                median = cms.double(-5.38205528259),
                upper = cms.double(-3.52304198265)
            ),
            pfcand_puppiw = cms.PSet(
                median = cms.double(1.0),
                upper = cms.double(1.0)
            ),
            pfcand_quality = cms.PSet(
                median = cms.double(0.0),
                upper = cms.double(5.0)
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
                upper = cms.double(1.2257443285)
            ),
            sv_costhetasvpv = cms.PSet(
                median = cms.double(0.999744534492),
                upper = cms.double(0.999992311001)
            ),
            sv_d3d = cms.PSet(
                median = cms.double(0.496477723122),
                upper = cms.double(4.83165483475)
            ),
            sv_d3dsig = cms.PSet(
                median = cms.double(6.50515079498),
                upper = cms.double(36.9069334412)
            ),
            sv_deltaR = cms.PSet(
                median = cms.double(0.105607174337),
                upper = cms.double(0.323846782446)
            ),
            sv_dxy = cms.PSet(
                median = cms.double(0.369336694479),
                upper = cms.double(3.97735537529)
            ),
            sv_dxysig = cms.PSet(
                median = cms.double(6.49603271484),
                upper = cms.double(36.8943783569)
            ),
            sv_erel_log = cms.PSet(
                median = cms.double(-2.77275466919),
                upper = cms.double(-1.7486346817)
            ),
            sv_etarel = cms.PSet(
                median = cms.double(-0.00318800867535),
                upper = cms.double(0.0951088288426)
            ),
            sv_mass = cms.PSet(
                median = cms.double(1.36728298664),
                upper = cms.double(4.75960615158)
            ),
            sv_normchi2 = cms.PSet(
                median = cms.double(0.796081960201),
                upper = cms.double(2.15187430382)
            ),
            sv_ntracks = cms.PSet(
                median = cms.double(3.0),
                upper = cms.double(5.0)
            ),
            sv_phirel = cms.PSet(
                median = cms.double(0.000510219600983),
                upper = cms.double(0.110185634494)
            ),
            sv_pt_log = cms.PSet(
                median = cms.double(3.77797603607),
                upper = cms.double(4.90166530609)
            ),
            sv_ptrel_log = cms.PSet(
                median = cms.double(-2.7622461319),
                upper = cms.double(-1.73727185726)
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
