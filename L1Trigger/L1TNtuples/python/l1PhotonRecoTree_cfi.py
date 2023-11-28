import FWCore.ParameterSet.Config as cms

l1PhotonRecoTree = cms.EDAnalyzer("L1PhotonRecoTreeProducer",
   maxPhoton                          = cms.uint32(20),
)

