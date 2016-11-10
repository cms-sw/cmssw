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

egmPhotonIsolation = cms.EDProducer( "CITKPFIsolationSumProducer",
                                     srcToIsolate = cms.InputTag("slimmedPhotons"),
                                     srcForIsolationCone = cms.InputTag('packedPFCandidates'),
                                     isolationConeDefinitions = IsoConeDefinitions
                                     )	

# The sequence defined here contains only one module. This is to keep the structure
# uniform with the AOD case where there are more modules in the analogous sequence.
egmPhotonIsolationMiniAODSequence = cms.Sequence( egmPhotonIsolation )

