import FWCore.ParameterSet.Config as cms
import PhysicsTools.IsolationAlgos.CITKPFIsolationSumProducerForPUPPI_cfi as _mod

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

egmElectronIsolationAODPUPPI = _mod.CITKPFIsolationSumProducerForPUPPI.clone(
                srcToIsolate = "gedGsfElectrons",
                srcForIsolationCone = '',
                isolationConeDefinitions = IsoConeDefinitions
)

egmElectronIsolationMiniAODPUPPI = _mod.CITKPFIsolationSumProducerForPUPPI.clone(
                srcToIsolate = "slimmedElectrons",
                srcForIsolationCone = 'packedPFCandidates',
                puppiValueMap = '',
                isolationConeDefinitions = IsoConeDefinitions
)

egmElectronIsolationMiniAODPUPPINoLeptons = _mod.CITKPFIsolationSumProducerForPUPPI.clone(
                srcToIsolate = "slimmedElectrons",
                srcForIsolationCone = 'packedPFCandidates',
                puppiValueMap = '',
                usePUPPINoLepton = True,
                isolationConeDefinitions = IsoConeDefinitions
)
