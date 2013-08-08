import FWCore.ParameterSet.Config as cms

process = cms.PSet()

process.fwliteInput = cms.PSet(
    fileNames = cms.vstring('makeROCcurveTauIdMVA.root'),
    
    maxEvents = cms.int32(-1),
    
    outputEvery = cms.uint32(1000)
)

process.showROCcurvesTauIdMVA = cms.PSet(

    graphs = cms.VPSet(
        cms.PSet(
            graphName = cms.string('hpsCombinedIsolation3HitsDeltaR04DB015'),
            legendEntry = cms.string("HPS 3hit: #DeltaR = 0.4, #Delta#beta = 0.15"),
            color = cms.int32(1)
        )
    ),

    xMin = cms.double(-0.05),
    xMax = cms.double(+1.05),
    yMin = cms.double(1.e-4),
    yMax = cms.double(1.e0),

    outputFileName = cms.string('showROCcurvesTauIdMVA.png')
)
