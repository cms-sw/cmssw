import FWCore.ParameterSet.Config as cms

egmGedGsfElectronPFIsolation = cms.EDProducer(
    "CITKPFIsolationSumProducer",
    srcToIsolate = cms.InputTag("gedGsfElectrons"),
    srcForIsolationCone = cms.InputTag('particleFlowTmp'),
    isolationConeDefinitions = cms.VPSet(
        cms.PSet( isolationAlgo = cms.string('ElectronPFIsolationWithConeVeto'), 
                  coneSize = cms.double(0.3),
                  VetoConeSizeBarrel = cms.double(0.0),
                  VetoConeSizeEndcaps = cms.double(0.015),
                  isolateAgainst = cms.string('h+'),
                  vertexSrc = cms.InputTag('offlinePrimaryVertices'),
                  miniAODVertexCodes = cms.vuint32(2,3) ),
        cms.PSet( isolationAlgo = cms.string('ElectronPFIsolationWithConeVeto'), 
                  coneSize = cms.double(0.3),
                  VetoConeSizeBarrel = cms.double(0.0),
                  VetoConeSizeEndcaps = cms.double(0.0),
                  vertexSrc = cms.InputTag('offlinePrimaryVertices'),
                  isolateAgainst = cms.string('h0'),
                  miniAODVertexCodes = cms.vuint32(2,3) ),
        cms.PSet( isolationAlgo = cms.string('ElectronPFIsolationWithConeVeto'), 
                  coneSize = cms.double(0.3),
                  VetoConeSizeBarrel = cms.double(0.0),
                  VetoConeSizeEndcaps = cms.double(0.08),
                  vertexSrc = cms.InputTag('offlinePrimaryVertices'),
                  isolateAgainst = cms.string('gamma'),
                  miniAODVertexCodes = cms.vuint32(2,3) ),
        cms.PSet( isolationAlgo = cms.string('ElectronPFIsolationWithConeVeto'), 
                  coneSize = cms.double(0.3),
                  VetoConeSizeBarrel = cms.double(0.0),
                  VetoConeSizeEndcaps = cms.double(0.015),
                  vertexSrc = cms.InputTag('offlinePrimaryVertices'),
                  isolateAgainst = cms.string('PUh+'),
                  miniAODVertexCodes = cms.vuint32(2,3) )
        )
    )

