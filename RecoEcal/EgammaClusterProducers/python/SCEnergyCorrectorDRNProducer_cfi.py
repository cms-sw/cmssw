import FWCore.ParameterSet.Config as cms

DRNProducerEB = cms.EDProducer('SCEnergyCorrectorDRNProducer',
   inputSCs = cms.InputTag('particleFlowSuperClusterECAL','particleFlowSuperClusterECALBarrel'),
   Client = cms.PSet(
        mode = cms.string("Async"),
        modelName = cms.string("MustacheEB"),
        modelConfigPath = cms.FileInPath("RecoEcal/EgammaClusterProducers/data/models/MustacheEB/config.pbtxt"),
        allowedTries = cms.untracked.uint32(1),
        timeout = cms.untracked.uint32(10),
    ),
 )


DRNProducerEE = cms.EDProducer('SCEnergyCorrectorDRNProducer',
   inputSCs = cms.InputTag('particleFlowSuperClusterECAL','particleFlowSuperClusterECALEndcapWithPreshower'),
   Client = cms.PSet(
        mode = cms.string("Async"),
        modelName = cms.string('MustacheEE'),
        modelConfigPath = cms.FileInPath("RecoEcal/EgammaClusterProducers/data/models/MustacheEE/config.pbtxt"),
        allowedTries = cms.untracked.uint32(1),
        timeout = cms.untracked.uint32(10),
    ),
 )


