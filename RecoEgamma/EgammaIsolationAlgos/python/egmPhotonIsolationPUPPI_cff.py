import FWCore.ParameterSet.Config as cms

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


egmPhotonIsolationAODPUPPI = cms.EDProducer( "CITKPFIsolationSumProducerForPUPPI",
			  srcToIsolate = cms.InputTag("gedPhotons"),
			  srcForIsolationCone = cms.InputTag('particleFlow'),
                          puppiValueMap = cms.InputTag('puppi'),
			  isolationConeDefinitions = IsoConeDefinitions
)

egmPhotonIsolationMiniAODPUPPI = egmPhotonIsolationAODPUPPI.clone()
egmPhotonIsolationMiniAODPUPPI.srcForIsolationCone = cms.InputTag("packedPFCandidates")
egmPhotonIsolationMiniAODPUPPI.srcToIsolate = cms.InputTag("slimmedPhotons")
egmPhotonIsolationMiniAODPUPPI.puppiValueMap = cms.InputTag('')
