import FWCore.ParameterSet.Config as cms

# Electron collection merger
mergedSlimmedElectronsForTauId = cms.EDProducer('PATElectronCollectionMerger',
    src = cms.VInputTag('slimmedElectrons', 'slimmedElectronsHGC')
)
