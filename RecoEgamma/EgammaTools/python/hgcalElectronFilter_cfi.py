import FWCore.ParameterSet.Config as cms

cleanedEcalDrivenGsfElectronsFromMultiCl = cms.EDProducer("HGCalElectronFilter",
        inputGsfElectrons = cms.InputTag("ecalDrivenGsfElectronsFromMultiCl"),
        outputCollection = cms.string(""),
        cleanBarrel = cms.bool(False)
        )
