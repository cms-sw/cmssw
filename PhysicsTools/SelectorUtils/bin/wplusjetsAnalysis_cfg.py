import FWCore.ParameterSet.Config as cms

process = cms.Process("FWLitePlots")


process.load('PhysicsTools.SelectorUtils.wplusjetsAnalysis_cfi')

process.wplusjetsAnalysis.muPlusJets = False
process.wplusjetsAnalysis.ePlusJets = True

process.wplusjetsAnalysis.muonSrc = cms.InputTag('cleanPatMuons')
process.wplusjetsAnalysis.electronSrc = cms.InputTag('cleanPatElectrons')
process.wplusjetsAnalysis.jetSrc = cms.InputTag('cleanPatJets')

#process.wplusjetsAnalysis.cutsToIgnore = cms.vstring( ['== 1 Tight Lepton, Mu Veto'] )

process.inputs = cms.PSet (
    fileNames = cms.vstring(
        # Your data goes here:
        'patTuple.root'
        )
)

process.outputs = cms.PSet (
    outputName = cms.string('wplusjetsPlots_mu.root')
)
