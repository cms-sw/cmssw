import FWCore.ParameterSet.Config as cms

hcalNoiseRates = cms.EDAnalyzer('HcalNoiseRates',
#    outputFile   = cms.untracked.string('NoiseRatesRelVal.root'),
    outputFile   = cms.untracked.string(''),
    rbxCollName  = cms.untracked.InputTag('hcalnoise'),
    minRBXEnergy = cms.untracked.double(20.0),
    minHitEnergy = cms.untracked.double(1.5),
    useAllHistos = cms.untracked.bool(False),
    noiselabel   = cms.InputTag('hcalnoise')
)
