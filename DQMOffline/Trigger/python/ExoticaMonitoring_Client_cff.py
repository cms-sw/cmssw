import FWCore.ParameterSet.Config as cms

metEfficiency = cms.EDAnalyzer("DQMGenericClient",
    subDirs        = cms.untracked.vstring("HLT/MET/*"),
    verbose        = cms.untracked.uint32(0), # Set to 2 for all messages
    resolution     = cms.vstring(),
    efficiency     = cms.vstring(
        "effic_met          'MET turnON;            PF MET [GeV]; efficiency'     met_numerator          met_denominator",
        "effic_met_variable 'MET turnON;            PF MET [GeV]; efficiency'     met_variable_numerator met_variable_denominator",
        "effic_metPhi       'MET efficiency vs phi; PF MET phi [rad]; efficiency' metPhi_numerator       metPhi_denominator",
    ),
    efficiencyProfile = cms.untracked.vstring(
        "effic_met_vs_LS 'MET efficiency vs LS; LS; PF MET efficiency' metVsLS_numerator metVsLS_denominator"
    ),
  
)

htEfficiency = cms.EDAnalyzer("DQMGenericClient",
    subDirs        = cms.untracked.vstring("HLT/Exotica/*"),
    verbose        = cms.untracked.uint32(0), # Set to 2 for all messages
    resolution     = cms.vstring(),
    efficiency     = cms.vstring(
        "effic_ht          'HT turnON;            PF HT [GeV]; efficiency'     ht_numerator          ht_denominator",
        "effic_ht_variable 'HT turnON;            PF HT [GeV]; efficiency'     ht_variable_numerator ht_variable_denominator",
    ),
    efficiencyProfile = cms.untracked.vstring(
        "effic_ht_vs_LS 'HT efficiency vs LS; LS; PF HT efficiency' htVsLS_numerator htVsLS_denominator"
    ),
  
)

exoticaClient = cms.Sequence(
    metEfficiency,
    htEfficiency
)
