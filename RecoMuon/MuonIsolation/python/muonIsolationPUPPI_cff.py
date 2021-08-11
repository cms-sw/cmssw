import FWCore.ParameterSet.Config as cms
import PhysicsTools.IsolationAlgos.CITKPFIsolationSumProducerForPUPPI_cfi as _mod

IsoConeDefinitions = cms.VPSet(
        cms.PSet( isolationAlgo = cms.string('MuonPFIsolationWithConeVeto'),
                  coneSize = cms.double(0.4),
                  VetoThreshold = cms.double(0.0),
                  VetoConeSize = cms.double(0.0001),# VetoConeSize is deltaR^2
                  isolateAgainst = cms.string('h+'),
                  miniAODVertexCodes = cms.vuint32(2,3) ),
        cms.PSet( isolationAlgo = cms.string('MuonPFIsolationWithConeVeto'),
                  coneSize = cms.double(0.4),
                  VetoThreshold = cms.double(0.0),
                  VetoConeSize = cms.double(0.01),# VetoConeSize is deltaR^2
                  isolateAgainst = cms.string('h0'),
                  miniAODVertexCodes = cms.vuint32(2,3) ),
        cms.PSet( isolationAlgo = cms.string('MuonPFIsolationWithConeVeto'),
                  coneSize = cms.double(0.4),
                  VetoThreshold = cms.double(0.0),
                  VetoConeSize = cms.double(0.01),# VetoConeSize is deltaR^2
                  isolateAgainst = cms.string('gamma'),
                  miniAODVertexCodes = cms.vuint32(2,3) ),                  
)

muonIsolationAODPUPPI = _mod.CITKPFIsolationSumProducerForPUPPI.clone(
                srcToIsolate = "muons",
                srcForIsolationCone = '',
                isolationConeDefinitions = IsoConeDefinitions
)

muonIsolationMiniAODPUPPI = _mod.CITKPFIsolationSumProducerForPUPPI.clone(
                srcToIsolate = "slimmedMuons",
                srcForIsolationCone = 'packedPFCandidates',
                puppiValueMap = '',
                isolationConeDefinitions = IsoConeDefinitions
)

muonIsolationMiniAODPUPPINoLeptons = _mod.CITKPFIsolationSumProducerForPUPPI.clone(
                srcToIsolate = "slimmedMuons",
                srcForIsolationCone = 'packedPFCandidates',
                puppiValueMap = '',
                usePUPPINoLepton = True,
                isolationConeDefinitions = IsoConeDefinitions
)
