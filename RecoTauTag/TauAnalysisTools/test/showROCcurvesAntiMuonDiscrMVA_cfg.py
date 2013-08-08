import FWCore.ParameterSet.Config as cms

process = cms.PSet()

process.fwliteInput = cms.PSet(
    fileNames = cms.vstring('makeROCcurveAntiMuonDiscrMVA.root'),
    
    maxEvents = cms.int32(-1),
    
    outputEvery = cms.uint32(1000)
)

process.showROCcurvesTauIdMVA = cms.PSet(

    graphs = cms.VPSet(
        cms.PSet(
            graphName = cms.string('antiMuonDiscrLoose2'),
            legendEntry = cms.string("anti-#mu Loose"),
            color = cms.int32(1)
        )
    ),

    xMin = cms.double(0.75),
    xMax = cms.double(1.05),
    yMin = cms.double(1.e-5),
    yMax = cms.double(1.e0),

    outputFileName = cms.string('showROCcurvesAntiMuonDiscrMVA.png')
)
