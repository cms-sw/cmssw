import FWCore.ParameterSet.Config as cms

qcdPhotonsDQM = cms.EDAnalyzer("QcdPhotonsDQM",
                # Event must pass this trigger
                triggerPathToPass         = cms.string("HLT_Photon20_Cleaned_L1R"),
                # Plot results of these triggers too (these don't *have* to be passed)
                plotTheseTriggersToo      = cms.vstring("HLT_Photon20_Cleaned_L1R","HLT_Photon30_Cleaned_L1R","HLT_Photon50_NoHE_Cleaned_L1R","HLT_DoublePhoton17_L1R"),
                # Collections
                hltMenu                   = cms.string("HLT"),
                triggerResultsCollection  = cms.string("TriggerResults"),
                photonCollection          = cms.InputTag("photons"),
                caloJetCollection         = cms.InputTag("ak5CaloJets"),
                vertexCollection          = cms.InputTag("offlinePrimaryVertices"),
                # Cuts on the reco objects
                minCaloJetPt              = cms.double(5.0),
                minPhotonEt               = cms.double(25.0),
                requirePhotonFound        = cms.bool(True),
                # Max Et on plots
                plotPhotonMaxEt           = cms.double(200.0),
                plotPhotonMaxEta          = cms.double(5.0),
                plotJetMaxEta             = cms.double(5.0)
)
