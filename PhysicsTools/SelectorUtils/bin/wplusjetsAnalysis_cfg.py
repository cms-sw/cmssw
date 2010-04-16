import FWCore.ParameterSet.Config as cms

process = cms.Process("FWLitePlots")


process.load('PhysicsTools.SelectorUtils.wplusjetsAnalysis_cfi')

process.inputs = cms.PSet (
    fileNames = cms.vstring(
        '/your/files/go/here.root'
        )
)

process.outputs = cms.PSet (
    outputName = cms.string('wplusjetsPlots.root')
)
