import FWCore.ParameterSet.Config as cms

qcdPhotonsDQM = cms.EDAnalyzer("QcdPhotonsDQM",
                # Event must pass this trigger
                triggerPathToPass        = cms.string("HLT_Photon15_L1R"),
                # Plot results of these triggers too (these don't *have* to be passed)
                plotTheseTriggersToo     = cms.vstring("HLT_Photon10_L1R","HLT_Photon15_L1R","HLT_Photon15_LooseEcalIso_L1R","HLT_Photon20_L1R","HLT_Photon30_L1R","HLT_Ele15_LW_L1R"),
                # Collections
                triggerResultsCollection = cms.InputTag("TriggerResults", "", "HLT"),
                photonCollection         = cms.InputTag("photons"),
                caloJetCollection        = cms.InputTag("sisCone5CaloJets"),
#               caloJetCollection        = cms.InputTag("L2L3CorJetSC5Calo"),
                # Cuts on the reco objects
                minCaloJetEt             = cms.int32(15),
                minPhotonEt              = cms.int32(20),
                requirePhotonFound       = cms.bool(True),
                # Max Et on plots
                plotMaxEt                = cms.double(500.0),
                plotPhotonMaxEta         = cms.double(5.0),
                plotJetMaxEta            = cms.double(5.0)
)
