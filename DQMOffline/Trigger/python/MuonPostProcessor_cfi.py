import FWCore.ParameterSet.Config as cms

hltMuonPostVal = cms.EDAnalyzer("DQMGenericClient",

    subDirs        = cms.untracked.vstring("HLT/Muon/Distributions/*"),
    verbose        = cms.untracked.uint32(0), # Set to 2 for all messages
    outputFileName = cms.untracked.string(''),
    commands       = cms.vstring(),
    resolution     = cms.vstring(),
    efficiency     = cms.vstring(),

    efficiencyProfile = cms.untracked.vstring(
        "efficiencyEta 'Efficiency to Match Reco Muons to Trigger Objects; #eta^{reco}; N(#mu matched to trigger object) / N(#mu)' efficiencyEta_numer efficiencyEta_denom",
        "efficiencyPhi 'Efficiency to Match Reco Muons to Trigger Objects; #phi^{reco}; N(#mu matched to trigger object) / N(#mu)' efficiencyPhi_numer efficiencyPhi_denom",
        "efficiencyTurnOn 'Efficiency to Match Reco Muons to Trigger Objects; p_{T}^{reco}; N(#mu matched to trigger object) / N(#mu)' efficiencyTurnOn_numer efficiencyTurnOn_denom",
        "efficiencyD0 'Efficiency to Match Reco Muons to Trigger Objects; d0^{reco}; N(#mu matched to trigger object) / N(#mu)' efficiencyD0_numer efficiencyD0_denom",
        "efficiencyZ0 'Efficiency to Match Reco Muons to Trigger Objects; z0^{reco}; N(#mu matched to trigger object) / N(#mu)' efficiencyZ0_numer efficiencyZ0_denom",
        "efficiencyCharge 'Efficiency to Match Reco Muons to Trigger Objects; q^{reco}; N(#mu matched to trigger object) / N(#mu)' efficiencyCharge_numer efficiencyCharge_denom",
        "fakerateEta 'Trigger Fake Rate; #eta^{trigger}; N(unmatched trigger objects) / N(trigger objects)' fakerateEta_numer fakerateEta_denom",
        "fakeratePhi 'Trigger Fake Rate; #phi^{trigger}; N(unmatched trigger objects) / N(trigger objects)' fakeratePhi_numer fakeratePhi_denom",
        "fakerateTurnOn 'Trigger Fake Rate; p_{T}^{trigger}; N(unmatched trigger objects) / N(trigger objects)' fakerateTurnOn_numer fakerateTurnOn_denom",
    )

)

