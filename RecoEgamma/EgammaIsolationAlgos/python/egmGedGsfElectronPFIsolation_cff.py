import FWCore.ParameterSet.Config as cms

egmGedGsfElectronPFNoPileUpIsolation = cms.EDProducer(
    "CITKPFIsolationSumProducer",
    srcToIsolate = cms.InputTag("gedGsfElectrons"),
    srcForIsolationCone = cms.InputTag('pfNoPileUpCandidates'),
    isolationConeDefinitions = cms.VPSet(
        cms.PSet( isolationAlgo = cms.string('ElectronPFIsolationWithConeVeto'), 
                  coneSize = cms.double(0.3),
                  VetoConeSizeBarrel = cms.double(0.0),
                  VetoConeSizeEndcaps = cms.double(0.015),
                  isolateAgainst = cms.string('h+'),
                  miniAODVertexCodes = cms.vuint32(2,3) ),
        cms.PSet( isolationAlgo = cms.string('ElectronPFIsolationWithConeVeto'), 
                  coneSize = cms.double(0.3),
                  VetoConeSizeBarrel = cms.double(0.0),
                  VetoConeSizeEndcaps = cms.double(0.0),
                  isolateAgainst = cms.string('h0'),
                  miniAODVertexCodes = cms.vuint32(2,3) ),
        cms.PSet( isolationAlgo = cms.string('ElectronPFIsolationWithConeVeto'), 
                  coneSize = cms.double(0.3),
                  VetoConeSizeBarrel = cms.double(0.0),
                  VetoConeSizeEndcaps = cms.double(0.08),
                  isolateAgainst = cms.string('gamma'),
                  miniAODVertexCodes = cms.vuint32(2,3) )
        )
    )

egmGedGsfElectronPFPileUpIsolation = cms.EDProducer(
    "CITKPFIsolationSumProducer",
    srcToIsolate = cms.InputTag("gedGsfElectrons"),
    srcForIsolationCone = cms.InputTag('pfPileUpAllChargedParticles'),
    isolationConeDefinitions = cms.VPSet(
        cms.PSet( isolationAlgo = cms.string('ElectronPFIsolationWithConeVeto'), 
                  coneSize = cms.double(0.3),
                  VetoConeSizeBarrel = cms.double(0.0),
                  VetoConeSizeEndcaps = cms.double(0.015),
                  isolateAgainst = cms.string('h+'),
                  miniAODVertexCodes = cms.vuint32(0,1) )
        )
    )



egmGedGsfElectronPFNoPileUpIsolationMapBasedVeto = cms.EDProducer(
    "CITKPFIsolationSumProducer",
    srcToIsolate = cms.InputTag("gedGsfElectrons"),
    srcForIsolationCone = cms.InputTag('pfNoPileUpCandidates'),
    isolationConeDefinitions = cms.VPSet(
         cms.PSet( isolationAlgo = cms.string('ElectronPFIsolationWithMapBasedVeto'),
                  coneSize = cms.double(0.3),
                  isolateAgainst = cms.string('h+'),
                  miniAODVertexCodes = cms.vuint32(2,3),
                  vertexIndex = cms.int32(0),
                  particleBasedIsolation = cms.InputTag("particleBasedIsolation", "gedGsfElectrons") ),
        cms.PSet( isolationAlgo = cms.string('ElectronPFIsolationWithMapBasedVeto'),
                  coneSize = cms.double(0.3),
                  isolateAgainst = cms.string('h0'),
                  miniAODVertexCodes = cms.vuint32(2,3),
                  vertexIndex = cms.int32(0),
                  particleBasedIsolation = cms.InputTag("particleBasedIsolation", "gedGsfElectrons") ),
        cms.PSet( isolationAlgo = cms.string('ElectronPFIsolationWithMapBasedVeto'),
                  coneSize = cms.double(0.3),
                  isolateAgainst = cms.string('gamma'),
                  miniAODVertexCodes = cms.vuint32(2,3),
                  vertexIndex = cms.int32(0),
                  particleBasedIsolation = cms.InputTag("particleBasedIsolation", "gedGsfElectrons") )
        )
  )


egmGedGsfElectronPFPileUpIsolationMapBasedVeto = cms.EDProducer(
    "CITKPFIsolationSumProducer",
    srcToIsolate = cms.InputTag("gedGsfElectrons"),
    srcForIsolationCone = cms.InputTag('pfPileUpAllChargedParticles'),
    isolationConeDefinitions = cms.VPSet(
        cms.PSet( isolationAlgo = cms.string('ElectronPFIsolationWithMapBasedVeto'),
                  coneSize = cms.double(0.3),
                  isolateAgainst = cms.string('h+'),
                  miniAODVertexCodes = cms.vuint32(2,3),
                  vertexIndex = cms.int32(0),
                  particleBasedIsolation = cms.InputTag("particleBasedIsolation", "gedGsfElectrons") )
          )
    )


