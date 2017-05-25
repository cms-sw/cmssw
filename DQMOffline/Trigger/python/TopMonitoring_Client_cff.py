import FWCore.ParameterSet.Config as cms

topEfficiency = cms.EDAnalyzer("DQMGenericClient",
    subDirs        = cms.untracked.vstring("HLT/TOP/*"),
    verbose        = cms.untracked.uint32(0), # Set to 2 for all messages
    resolution     = cms.vstring(),
    efficiency     = cms.vstring(
        "effic_muPhi       'mu efficiency vs phi; muon phi ; efficiency' muPhi_numerator       muPhi_denominator",
        "effic_muEta       'mu efficiency vs eta; muon eta ; efficiency' muEta_numerator       muEta_denominator",
        "effic_muPt       'mu efficiency vs pt; muon pt [GeV]; efficiency' muPt_numerator       muPt_denominator",
        "effic_elePhi       'electron efficiency vs phi; electron phi ; efficiency' elePhi_numerator       elePhi_denominator",
        "effic_eleEta       'electron efficiency vs eta; electron eta ; efficiency' eleEta_numerator       eleEta_denominator",
        "effic_elePt       'electron efficiency vs pt; electron pt [GeV]; efficiency' elePt_numerator       elePt_denominator",
        "effic_jetPhi       'jet efficiency vs phi; jet phi ; efficiency' jetPhi_numerator       jetPhi_denominator",
        "effic_jetEta       'jet efficiency vs eta; jet eta ; efficiency' jetEta_numerator       jetEta_denominator",
        "effic_jetPt       'jet efficiency vs pt; jet pt [GeV]; efficiency' jetPt_numerator       jetPt_denominator",
    ),
)

topClient = cms.Sequence(
    topEfficiency
)
