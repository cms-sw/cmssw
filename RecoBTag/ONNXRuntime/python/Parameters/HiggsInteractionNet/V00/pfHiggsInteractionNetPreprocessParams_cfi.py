import FWCore.ParameterSet.Config as cms

pfHiggsInteractionNetPreprocessParams = cms.PSet(
    input_names = cms.vstring(
        'input_cpf',
        'input_sv'
    ),
    input_cpf = cms.PSet(
        input_shape = cms.vuint32(1, 30, 60),
        var_length = cms.uint32(60),
        pad = cms.double(0),
        var_names = cms.vstring(
            'cpf_ptrel', 
            'cpf_erel', 
            'cpf_phirel', 
            'cpf_etarel', 
            'cpf_deltaR', 
            'cpf_drminsv', 
            'cpf_drsubjet1', 
            'cpf_drsubjet2', 
            'cpf_dz', 
            'cpf_dzsig', 
            'cpf_dxy', 
            'cpf_dxysig', 
            'cpf_normchi2', 
            'cpf_quality', 
            'cpf_dptdpt', 
            'cpf_detadeta', 
            'cpf_dphidphi', 
            'cpf_dxydxy', 
            'cpf_dzdz', 
            'cpf_dxydz', 
            'cpf_dphidxy', 
            'cpf_dlambdadz', 
            'cpf_btagEtaRel', 
            'cpf_btagPtRatio', 
            'cpf_btagPParRatio', 
            'cpf_btagSip2dVal', 
            'cpf_btagSip2dSig', 
            'cpf_btagSip3dVal', 
            'cpf_btagSip3dSig', 
            'cpf_btagJetDistVal'
        )
    ),
    input_sv = cms.PSet(
        input_shape = cms.vuint32(1, 14, 5),
        var_length = cms.uint32(5),
        pad = cms.double(0),
        var_names = cms.vstring(
            'sv_ptrel', 
            'sv_erel', 
            'sv_phirel', 
            'sv_etarel', 
            'sv_deltaR',
            'sv_pt',  
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
