import FWCore.ParameterSet.Config as cms

qcdPhotonsDQM = cms.EDAnalyzer("QcdPhotonsDQM",
                # Event must pass this trigger
                triggerPathToPass         = cms.string("HLT_Photon15_L1R"),
                # Plot results of these triggers too (these don't *have* to be passed)
                plotTheseTriggersToo      = cms.vstring("HLT_Photon10_L1R","HLT_Photon10_LooseEcalIso_TrackIso_L1R","HLT_Photon15_L1R","HLT_Photon25_LooseEcalIso_TrackIso_L1R","HLT_Photon25_L1R","HLT_Photon30_L1R_1E31","HLT_Ele15_SW_L1R"),
                # Collections
                hltMenu                   = cms.string("HLT"),
                triggerResultsCollection  = cms.string("TriggerResults"),
                photonCollection          = cms.InputTag("photons"),
                caloJetCollection         = cms.InputTag("ak5CaloJets"),
#               caloJetCollection         = cms.InputTag("sisCone5CaloJets"),
                # Cuts on the reco objects
                minCaloJetEt              = cms.int32(15),
                minPhotonEt               = cms.int32(20),
                requirePhotonFound        = cms.bool(True),
                # Max Et on plots
                plotMaxEt                 = cms.double(500.0),
                plotPhotonMaxEta          = cms.double(5.0),
                plotJetMaxEta             = cms.double(5.0)
)
