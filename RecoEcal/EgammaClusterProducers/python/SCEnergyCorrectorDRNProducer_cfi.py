import FWCore.ParameterSet.Config as cms

DRNProducerEB = cms.EDProducer('SCEnergyCorrectorDRNProducer',
   correctorCfg = cms.PSet(
     ecalRecHitsEE = cms.InputTag('reducedEcalRecHitsEE'),
     ecalRecHitsEB = cms.InputTag('reducedEcalRecHitsEB'),
     rhoFastJet = cms.InputTag("fixedGridRhoAll"),
   ),
   inputSCs = cms.InputTag('particleFlowSuperClusterECAL','particleFlowSuperClusterECALBarrel'),
   Client = cms.PSet(
       mode = cms.string("Async"),
        preferredServer = cms.untracked.string(""),
        timeout = cms.untracked.uint32(10),
        modelName = cms.string("MustacheEB"),
        modelVersion = cms.string(""),
        modelConfigPath = cms.FileInPath("DRNData/models/MustacheEB/config.pbtxt"),
        verbose = cms.untracked.bool(False),
        allowedTries = cms.untracked.uint32(1),
        useSharedMemory = cms.untracked.bool(True),
        compression = cms.untracked.string(""),
    ),
 )


DRNProducerEE = cms.EDProducer('SCEnergyCorrectorDRNProducer',
   correctorCfg = cms.PSet(
     ecalRecHitsEE = cms.InputTag('reducedEcalRecHitsEE'),
     ecalRecHitsEB = cms.InputTag('reducedEcalRecHitsEB'),
     rhoFastJet = cms.InputTag("fixedGridRhoAll"),
   ),
   inputSCs = cms.InputTag('particleFlowSuperClusterECAL','particleFlowSuperClusterECALEndcapWithPreshower'),
   Client = cms.PSet(
        mode = cms.string("Async"),
        preferredServer = cms.untracked.string(""),
        timeout = cms.untracked.uint32(10),
        modelName = cms.string('MustacheEE'),
        modelVersion = cms.string(""),
        modelConfigPath = cms.FileInPath("DRNData/models/MustacheEE/config.pbtxt"),
        verbose = cms.untracked.bool(False),
        allowedTries = cms.untracked.uint32(1),
        useSharedMemory = cms.untracked.bool(True),
        compression = cms.untracked.string(""),
    ),
 )


