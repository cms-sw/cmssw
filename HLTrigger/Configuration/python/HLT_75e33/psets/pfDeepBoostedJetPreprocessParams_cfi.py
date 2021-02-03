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
                norm_factor = cms.double(1.0)
            ),
            pfcand_abseta = cms.PSet(
                median = cms.double(0.632648706436),
                norm_factor = cms.double(1.59032225958)
            ),
            pfcand_btagEtaRel = cms.PSet(
                median = cms.double(1.19703966379),
                norm_factor = cms.double(0.521026991705)
            ),
            pfcand_btagJetDistVal = cms.PSet(
                median = cms.double(-0.000215483247302),
                norm_factor = cms.double(161.385119349)
            ),
            pfcand_btagPParRatio = cms.PSet(
                median = cms.double(0.839023888111),
                norm_factor = cms.double(1.19186117841)
            ),
            pfcand_btagPtRatio = cms.PSet(
                median = cms.double(0.0173742230982),
                norm_factor = cms.double(4.25351138308)
            ),
            pfcand_btagSip2dSig = cms.PSet(
                median = cms.double(0.0),
                norm_factor = cms.double(1.41174531059)
            ),
            pfcand_btagSip2dVal = cms.PSet(
                median = cms.double(0.0),
                norm_factor = cms.double(495.583709284)
            ),
            pfcand_btagSip3dSig = cms.PSet(
                median = cms.double(0.0),
                norm_factor = cms.double(0.831133090749)
            ),
            pfcand_btagSip3dVal = cms.PSet(
                median = cms.double(0.0),
                norm_factor = cms.double(233.664322627)
            ),
            pfcand_charge = cms.PSet(
                median = cms.double(0.0),
                norm_factor = cms.double(1.0)
            ),
            pfcand_deltaR = cms.PSet(
                median = cms.double(0.252654820681),
                norm_factor = cms.double(3.50836328292)
            ),
            pfcand_detadeta = cms.PSet(
                median = cms.double(1.15567200254e-08),
                norm_factor = cms.double(1644010.14927)
            ),
            pfcand_dlambdadz = cms.PSet(
                median = cms.double(-1.2810873784e-07),
                norm_factor = cms.double(268715.13012)
            ),
            pfcand_dphidphi = cms.PSet(
                median = cms.double(1.130510352e-08),
                norm_factor = cms.double(796482.476472)
            ),
            pfcand_dphidxy = cms.PSet(
                median = cms.double(-9.61654578191e-08),
                norm_factor = cms.double(204149.346943)
            ),
            pfcand_dptdpt = cms.PSet(
                median = cms.double(4.11880840545e-08),
                norm_factor = cms.double(66429.1000843)
            ),
            pfcand_drminsv = cms.PSet(
                median = cms.double(0.192182734609),
                norm_factor = cms.double(2.38205282141)
            ),
            pfcand_drsubjet1 = cms.PSet(
                median = cms.double(0.258241385221),
                norm_factor = cms.double(2.92607580997)
            ),
            pfcand_drsubjet2 = cms.PSet(
                median = cms.double(0.305164307356),
                norm_factor = cms.double(2.88195895791)
            ),
            pfcand_dxy = cms.PSet(
                median = cms.double(0.0),
                norm_factor = cms.double(415.415835966)
            ),
            pfcand_dxydxy = cms.PSet(
                median = cms.double(9.32332170578e-07),
                norm_factor = cms.double(45949.2394216)
            ),
            pfcand_dxydz = cms.PSet(
                median = cms.double(0.0),
                norm_factor = cms.double(10598589.4298)
            ),
            pfcand_dxysig = cms.PSet(
                median = cms.double(0.0),
                norm_factor = cms.double(1.54565964258)
            ),
            pfcand_dz = cms.PSet(
                median = cms.double(0.0),
                norm_factor = cms.double(264.770519024)
            ),
            pfcand_dzdz = cms.PSet(
                median = cms.double(1.58729170607e-06),
                norm_factor = cms.double(36545.958354)
            ),
            pfcand_dzsig = cms.PSet(
                median = cms.double(0.0),
                norm_factor = cms.double(1.23840120847)
            ),
            pfcand_erel_log = cms.PSet(
                median = cms.double(-5.38221979141),
                norm_factor = cms.double(0.556499386531)
            ),
            pfcand_etarel = cms.PSet(
                median = cms.double(-0.00711047858931),
                norm_factor = cms.double(4.2642743837)
            ),
            pfcand_hcalFrac = cms.PSet(
                median = cms.double(0.0),
                norm_factor = cms.double(1.0)
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
                median = cms.double(2.0),
                norm_factor = cms.double(0.00100300902708)
            ),
            pfcand_phirel = cms.PSet(
                median = cms.double(-1.71389010575e-05),
                norm_factor = cms.double(4.22784626632)
            ),
            pfcand_pt_log = cms.PSet(
                median = cms.double(1.03476798534),
                norm_factor = cms.double(0.542224410728)
            ),
            pfcand_ptrel_log = cms.PSet(
                median = cms.double(-5.37407469749),
                norm_factor = cms.double(0.554677114485)
            ),
            pfcand_puppiw = cms.PSet(
                median = cms.double(1.0),
                norm_factor = cms.double(255.000015199)
            ),
            pfcand_quality = cms.PSet(
                median = cms.double(5.0),
                norm_factor = cms.double(0.2)
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
                median = cms.double(0.616221785545),
                norm_factor = cms.double(1.49676942133)
            ),
            sv_costhetasvpv = cms.PSet(
                median = cms.double(0.999747157097),
                norm_factor = cms.double(174.907183727)
            ),
            sv_d3d = cms.PSet(
                median = cms.double(0.5242870152),
                norm_factor = cms.double(0.255813267634)
            ),
            sv_d3dsig = cms.PSet(
                median = cms.double(9.12465429306),
                norm_factor = cms.double(0.0238374692882)
            ),
            sv_deltaR = cms.PSet(
                median = cms.double(0.140969499946),
                norm_factor = cms.double(4.30546783192)
            ),
            sv_dxy = cms.PSet(
                median = cms.double(0.387232214212),
                norm_factor = cms.double(0.360931771841)
            ),
            sv_dxysig = cms.PSet(
                median = cms.double(9.11130714417),
                norm_factor = cms.double(0.0238327380073)
            ),
            sv_erel_log = cms.PSet(
                median = cms.double(-2.82667005062),
                norm_factor = cms.double(0.704463981589)
            ),
            sv_etarel = cms.PSet(
                median = cms.double(-0.0037129354896),
                norm_factor = cms.double(6.99426943996)
            ),
            sv_mass = cms.PSet(
                median = cms.double(1.34319722652),
                norm_factor = cms.double(0.368495534421)
            ),
            sv_normchi2 = cms.PSet(
                median = cms.double(0.819934427738),
                norm_factor = cms.double(0.725797320076)
            ),
            sv_ntracks = cms.PSet(
                median = cms.double(3.0),
                norm_factor = cms.double(0.5)
            ),
            sv_phirel = cms.PSet(
                median = cms.double(0.000521215377375),
                norm_factor = cms.double(7.16761972364)
            ),
            sv_pt_log = cms.PSet(
                median = cms.double(3.64881515503),
                norm_factor = cms.double(0.725050067872)
            ),
            sv_ptrel_log = cms.PSet(
                median = cms.double(-2.81496477127),
                norm_factor = cms.double(0.701236308041)
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