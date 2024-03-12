import FWCore.ParameterSet.Config as cms

# Electron collection merger
mergedSlimmedElectronsForTauId = cms.EDProducer('PATElectronCollectionMerger',
    src = cms.VInputTag('slimmedElectrons', 'slimmedElectronsHGC')
)
# foo bar baz
# 1GLUzRo7WNj66
