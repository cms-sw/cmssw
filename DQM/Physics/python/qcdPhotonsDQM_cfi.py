import FWCore.ParameterSet.Config as cms

qcdPhotonsDQM = cms.EDAnalyzer("QcdPhotonsDQM",
                            triggerPathToPass        = cms.string("HLT_Photon15_L1R"),
                            triggerResultsCollection = cms.InputTag("TriggerResults", "", "HLT"),
                            photonCollection         = cms.InputTag("photons"),
                            caloJetCollection        = cms.InputTag("sisCone5CaloJets"),
#                           caloJetCollection        = cms.InputTag("L2L3CorJetSC5Calo"),
                            minCaloJetEt             = cms.int32(15),
                            minPhotonEt              = cms.int32(20)
)
