import FWCore.ParameterSet.Config as cms

# DQM monitor module for QCD Photons:
#qcdPhotonsDQM = cms.EDAnalyzer("QcdPhotonsDQM",
#                                    photonCollection = cms.InputTag("photons"),
#)
qcdPhotonsDQM = cms.EDAnalyzer("QcdPhotonsDQM",
                            triggerPathToPass        = cms.string("HLT_Photon15_L1R"),
                            triggerResultsCollection = cms.InputTag("TriggerResults", "", "HLT"),
                            photonCollection         = cms.InputTag("photons"),
                            caloJetCollection        = cms.InputTag("sisCone5CaloJets"),
#                           caloJetCollection        = cms.InputTag("L2L3CorJetSC5Calo"),
)
