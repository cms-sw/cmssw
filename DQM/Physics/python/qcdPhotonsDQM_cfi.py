import FWCore.ParameterSet.Config as cms

qcdPhotonsDQM = cms.EDAnalyzer("QcdPhotonsDQM",
                # Event must pass this trigger
                triggerPathToPass         = cms.string("HLT_Photon"),
                # Plot results of these triggers too (these don't *have* to be passed)
                plotTheseTriggersToo      = cms.vstring("HLT_Photon20","HLT_Photon25","HLT_Photon30","HLT_Photon50","HLT_DoublePhoton"),
                # Collections
                trigTag                   = cms.untracked.InputTag("TriggerResults::HLT"),
                photonCollection          = cms.InputTag("gedPhotons"),
                jetCollection             = cms.InputTag("ak4PFJets"),
                vertexCollection          = cms.InputTag("offlinePrimaryVertices"),
                # Cuts on the reco objects
                minJetPt                  = cms.double(5.0),
                minPhotonEt               = cms.double(25.0),
                requirePhotonFound        = cms.bool(True),
                # Max Et on plots
                plotPhotonMaxEt           = cms.double(200.0),
                plotPhotonMaxEta          = cms.double(5.0),
                plotJetMaxEta             = cms.double(5.0),
                barrelRecHitTag           = cms.InputTag("reducedEcalRecHitsEB"),
                endcapRecHitTag           = cms.InputTag("reducedEcalRecHitsEE")
)
