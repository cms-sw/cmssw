import FWCore.ParameterSet.Config as cms

pileupVertexFilter  = cms.EDFilter('PAPileUpVertexFilter',
    vtxSrc = cms.InputTag("offlinePrimaryVertices"),
    doDzNtrkCut = cms.bool(True),
    doDxyDzCut = cms.bool(False),
    doSurfaceCut = cms.bool(False),
    dxyVeto = cms.double(999.),
    dzVeto = cms.double(-999.),
    dxyDzCutPar0 = cms.double(0.6),
    dxyDzCutPar1 = cms.double(13.333),
    surfaceMinDzEval = cms.double(0.2),
    dzCutByNtrk = cms.vdouble(
        999., 3.0, 2.4, 2.0, 1.2, 1.2, 0.9, 0.6
    ),
    surfaceFunctionString = cms.string("[2]*exp(-x**2/[0])*x**[3]+[1]+([6]*exp(-x/[4])*x**[7]+[5])*(y-[10]*exp(-x**2/[8])*x**[11]-[9])*(y-[10]*exp(-x**2/[8])*x**[11]-[9])"),
    surfaceCutParameters = cms.vdouble(
       0.924730, 7.484908, 8.849780, -0.587169,
       0.478601, -0.000106, -0.000385, -0.094790,
       0.250266, 198.662432, 728.424750, 2.958134  
    )
)
