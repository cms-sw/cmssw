import FWCore.ParameterSet.Config as cms

tag_conversion = 'allConversions'
conv_algo = 'undefined'
conv_qual = ['highPurity','generalTracksOnly']
tag_primary_vertex = 'offlinePrimaryVertices'
conv_vertex_rho = 1.5
conv_vtx_comp = False
conv_tk_vtx = 5
conv_inn_hits = True
conv_min_dof = 3
conv_high_purity = False
tag_pfCandidates = 'particleFlow'
pi0_online_switch = False
pi0_small_min = 0.130
pi0_small_max = 0.140
pi0_large_min = 0.110
pi0_large_max = 0.160

PhotonCandidates = cms.EDProducer('OniaPhotonConversionProducer',
    conversions = cms.InputTag(tag_conversion),
    convAlgo    = cms.string(conv_algo),
    convQuality = cms.vstring(conv_qual),
    primaryVertexTag = cms.InputTag(tag_primary_vertex),
    convSelection = cms.string('conversionVertex.position.rho>{0}'.format(conv_vertex_rho) ),
    wantTkVtxCompatibility = cms.bool(conv_vtx_comp),
    sigmaTkVtxComp = cms.uint32(conv_tk_vtx),
    wantCompatibleInnerHits = cms.bool(conv_inn_hits),
    pfcandidates = cms.InputTag(tag_pfCandidates),
    pi0OnlineSwitch = cms.bool(pi0_online_switch),
    pi0SmallWindow   = cms.vdouble(pi0_small_min, pi0_small_max),
    pi0LargeWindow   = cms.vdouble(pi0_large_min, pi0_large_max),
    TkMinNumOfDOF = cms.uint32(conv_min_dof),
    wantHighpurity = cms.bool(conv_high_purity),
    vertexChi2ProbCut = cms.double(0.0005),
    trackchi2Cut = cms.double(10),
    minDistanceOfApproachMinCut = cms.double(-0.25),
    minDistanceOfApproachMaxCut = cms.double(1.00)
    )
