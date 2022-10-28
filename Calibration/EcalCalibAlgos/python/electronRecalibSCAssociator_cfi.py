import FWCore.ParameterSet.Config as cms

electronRecalibSCAssociator = cms.EDProducer("ElectronRecalibSuperClusterAssociator",
                                             superClusterCollectionEB = cms.InputTag("particleFlowSuperClusterECAL:particleFlowSuperClusterECALBarrel"),
                                             superClusterCollectionEE = cms.InputTag("particleFlowSuperClusterECAL:particleFlowSuperClusterECALEndcapWithPreshower"),
    electronSrc = cms.InputTag('gedGsfElectrons'),
    outputLabel = cms.string('recalibSC')
)
