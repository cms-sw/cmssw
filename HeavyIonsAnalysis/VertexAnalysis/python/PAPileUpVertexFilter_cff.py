import FWCore.ParameterSet.Config as cms

import HeavyIonsAnalysis.VertexAnalysis.PAPileUpVertexFilter_cfi

pileupVertexFilterCutG = HeavyIonsAnalysis.VertexAnalysis.PAPileUpVertexFilter_cfi.pileupVertexFilter.clone()

pileupVertexFilterCutGloose = pileupVertexFilterCutG.clone(
    dzCutByNtrk = cms.vdouble(
        999., 4.5, 3.2, 3.0, 1.8, 1.8, 1.35, 0.9
    )
)

pileupVertexFilterCutGtight = pileupVertexFilterCutG.clone(
    dzCutByNtrk = cms.vdouble(
        999., 2.0, 1.6, 1.333, 0.8, 0.8, 0.6, 0.4
    )
)

pileupVertexFilterCutGplus = pileupVertexFilterCutG.clone(
    dzCutByNtrk = cms.vdouble(
        999., 999., 999., 3.0, 2.0,
        1.6, 1.4, 1.2, 1.1, 1.0,
        0.9, 0.8, 0.7, 0.7, 0.6,
        0.6, 0.5, 0.5, 0.4, 0.4,
        0.4, 0.3, 0.3, 0.3, 0.3, 
        0.3, 0.2, 0.2, 0.2, 0.2, 
        0.0
    ),
    dxyVeto = cms.double(0.05)
)

pileupVertexFilterCutGplusNV = pileupVertexFilterCutG.clone(
    dzCutByNtrk = cms.vdouble(
        999., 999., 999., 3.0, 2.0,
        1.6, 1.4, 1.2, 1.1, 1.0,
        0.9, 0.8, 0.7, 0.7, 0.6,
        0.6, 0.5, 0.5, 0.4, 0.4,
        0.4, 0.3, 0.3, 0.3, 0.3,
        0.3, 0.2, 0.2, 0.2, 0.2,
        0.0
    ),
    dxyVeto = cms.double(999.)
)


pileupVertexFilterCutE = pileupVertexFilterCutG.clone(
    doDzNtrkCut = cms.bool(False),
    doDxyDzCut = cms.bool(True)
)

pileupVertexFilterCutEandG = pileupVertexFilterCutG.clone(
    doDzNtrkCut = cms.bool(True),
    doDxyDzCut = cms.bool(True)
)

pileupVertexFilterCutW = pileupVertexFilterCutG.clone(
    doDzNtrkCut = cms.bool(False),
    doSurfaceCut = cms.bool(True)
)

pileupVertexFilterCutWplus = pileupVertexFilterCutG.clone(
    doDzNtrkCut = cms.bool(False),
    doSurfaceCut = cms.bool(True),
    dxyVeto = cms.double(0.05)
)  


pileupVertexFilterCutGplusplus = pileupVertexFilterCutG.clone(
    doDzNtrkCut = cms.bool(False),
    doSurfaceCut = cms.bool(True),
    dxyVeto = cms.double(0.05),
    surfaceMinDzEval = cms.double(0.0),
    surfaceFunctionString = cms.string("[0]*exp([1]*(x-([3]*(y-[4])**2+[5])))+[2]"),
    surfaceCutParameters = cms.vdouble(
    28.593,-1.525,2.636788,-1.5e-5,200.,0.0
    ) 
)

pileupVertexFilterCutGplusUpsPP = pileupVertexFilterCutG.clone(
    dzCutByNtrk = cms.vdouble(
        999., 999., 1.5, 1.0, 0.8,
        0.6, 0.5, 0.4, 0.3, 0.2,
        0.2, 0.2, 0.2, 0.2, 0.2,
        0.2, 0.0
    ),
    dxyVeto = cms.double(999.)
)

