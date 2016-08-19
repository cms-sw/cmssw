import FWCore.ParameterSet.Config as cms

IsoConeDefinitions = cms.VPSet(
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

egmElectronIsolationAODPUPPI = cms.EDProducer( "CITKPFIsolationSumProducer",
							  srcToIsolate = cms.InputTag("gedGsfElectrons"),
							  srcForIsolationCone = cms.InputTag('puppi'),
							  puppiValueMap = cms.InputTag(''),
							  isolationConeDefinitions = IsoConeDefinitions
)

egmElectronIsolationMiniAODPUPPI = egmPhotonIsolationAODPUPPI.clone()
egmElectronIsolationMiniAODPUPPI.srcForIsolationCone = cms.InputTag("packedPFCandidates")
egmElectronIsolationMiniAODPUPPI.srcToIsolate = cms.InputTag("slimmedElectrons")
egmElectronIsolationMiniAODPUPPI.puppiValueMap = cms.InputTag('')
