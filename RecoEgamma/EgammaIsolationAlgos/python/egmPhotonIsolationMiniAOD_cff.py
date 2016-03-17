import FWCore.ParameterSet.Config as cms

IsoConeDefinitions = cms.VPSet(cms.PSet( isolationAlgo = cms.string('PhotonPFIsolationWithMapBasedVeto'),
                                      coneSize = cms.double(0.3),
                                      isolateAgainst = cms.string('h+'),
                                      miniAODVertexCodes = cms.vuint32(2,3),
                                      vertexIndex = cms.int32(0),
                                    ),
                               cms.PSet( isolationAlgo = cms.string('PhotonPFIsolationWithMapBasedVeto'),
                                      coneSize = cms.double(0.3),
                                      isolateAgainst = cms.string('h0'),
                                      miniAODVertexCodes = cms.vuint32(2,3),
                                      vertexIndex = cms.int32(0),
                                    ),
                               cms.PSet( isolationAlgo = cms.string('PhotonPFIsolationWithMapBasedVeto'),
                                      coneSize = cms.double(0.3),
                                      isolateAgainst = cms.string('gamma'),
                                      miniAODVertexCodes = cms.vuint32(2,3),
                                      vertexIndex = cms.int32(0),
                                    )
    )

egmPhotonIsolationMiniAOD = cms.EDProducer( "CITKPFIsolationSumProducer",
			  srcToIsolate = cms.InputTag("slimmedPhotons"),
			  srcForIsolationCone = cms.InputTag('packedPFCandidates'),
			  isolationConeDefinitions = IsoConeDefinitions
  )	
