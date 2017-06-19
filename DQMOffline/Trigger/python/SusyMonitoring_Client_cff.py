import FWCore.ParameterSet.Config as cms

metSOSEfficiency = cms.EDAnalyzer("DQMGenericClient",
    subDirs        = cms.untracked.vstring("HLT/SOS/*"),
    verbose        = cms.untracked.uint32(0), # Set to 2 for all messages
    resolution     = cms.vstring(),
    efficiency     = cms.vstring(
        "effic_met          'MET turnON;            PF MET [GeV]; efficiency'     met_numerator          met_denominator",
 "efficPhi_met          'MET efficiency phi;            Phi[rad];efficiency'     met_Phi_numerator          met_Phi_denominator", 
"effic_ht          'HT turnON;            HT [GeV]; efficiency'     metht_numerator          metht_denominator",  
 ),
efficiencyProfile = cms.untracked.vstring(
        "effic_met_vs_LS 'MET efficiency vs LS; LS; PF MET efficiency' metVsLS_numerator metVsLS_denominator"
),
)



muSOSEfficiency = cms.EDAnalyzer("DQMGenericClient",
    subDirs        = cms.untracked.vstring("HLT/SOS/*"),
    verbose        = cms.untracked.uint32(0), # Set to 2 for all messages
    resolution     = cms.vstring(),
    efficiency     = cms.vstring(
  "effic_mu          'MU turnON;            PT(mu) [GeV]; efficiency'     mu_numerator          mu_denominator",
 "efficPhiEta_mu          'MU efficiency phi-eta;            Phi[rad]; Eta'     mu_PhiEta_numerator          mu_PhiEta_denominator",
    ),
efficiencyProfile = cms.untracked.vstring(
        "effic_mu_vs_LS 'MU efficiency vs LS; LS; PT(mu) efficiency' muVsLS_numerator muVsLS_denominator"
),
)


susyClient = cms.Sequence(
metSOSEfficiency
#+muSOSEfficiency
)
