import FWCore.ParameterSet.Config as cms
import PhysicsTools.IsolationAlgos.CITKPFIsolationSumProducerForPUPPI_cfi as _mod

IsoConeDefinitions = cms.VPSet(cms.PSet( isolationAlgo = cms.string('PhotonPFIsolationWithMapBasedVeto'),
                                      coneSize = cms.double(0.3),
                                      isolateAgainst = cms.string('h+'),
                                      miniAODVertexCodes = cms.vuint32(2,3),
                                      vertexIndex = cms.int32(0),
                                      particleBasedIsolation = cms.InputTag("reducedEgamma", "reducedPhotonPfCandMap"),

                                    ),
                               cms.PSet( isolationAlgo = cms.string('PhotonPFIsolationWithMapBasedVeto'),
                                      coneSize = cms.double(0.3),
                                      isolateAgainst = cms.string('h0'),
                                      miniAODVertexCodes = cms.vuint32(2,3),
                                      vertexIndex = cms.int32(0),
                                      particleBasedIsolation = cms.InputTag("reducedEgamma", "reducedPhotonPfCandMap"),
                                    ),
                               cms.PSet( isolationAlgo = cms.string('PhotonPFIsolationWithMapBasedVeto'),
                                      coneSize = cms.double(0.3),
                                      isolateAgainst = cms.string('gamma'),
                                      miniAODVertexCodes = cms.vuint32(2,3),
                                      vertexIndex = cms.int32(0),
                                      particleBasedIsolation = cms.InputTag("reducedEgamma", "reducedPhotonPfCandMap"),
                                    )
    )


egmPhotonIsolationAODPUPPI = _mod.CITKPFIsolationSumProducerForPUPPI.clone(
			  srcToIsolate = "gedPhotons",
			  srcForIsolationCone = 'particleFlow',
			  isolationConeDefinitions = IsoConeDefinitions
)

egmPhotonIsolationMiniAODPUPPI = egmPhotonIsolationAODPUPPI.clone(
                          srcForIsolationCone = "packedPFCandidates",
                          srcToIsolate        = "slimmedPhotons",
                          puppiValueMap       = ''
)
