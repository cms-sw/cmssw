import FWCore.ParameterSet.Config as cms
from DQMServices.Core.DQMEDHarvester import DQMEDHarvester

SoftMuHardJetMETEfficiency_muPt = DQMEDHarvester("DQMGenericClient",
    subDirs        = cms.untracked.vstring("HLT/SUSY/SoftMuHardJetMET/Muon"),
    verbose        = cms.untracked.uint32(0),
    resolution     = cms.vstring(),
    efficiency     = cms.vstring(
        "effic_muPt  'Efficiency vs Muon p_{T} ;  Muon p_{T} [GeV] ; efficiency'  muPt_1_variableBinning_numerator  muPt_1_variableBinning_denominator",
        "effic_muEta 'Efficiency vs Muon #eta ; Muon #eta ;          efficiency' muEta_1_variableBinning_numerator muEta_1_variableBinning_denominator",
    ),
)

SoftMuHardJetMETEfficiency_jetPt = DQMEDHarvester("DQMGenericClient",
    subDirs        = cms.untracked.vstring("HLT/SUSY/SoftMuHardJetMET/Jet"),
    verbose        = cms.untracked.uint32(0),
    resolution     = cms.vstring(),
    efficiency     = cms.vstring(
        "effic_jetPt  'Efficiency vs Jet p_{T} ; PFJet p_{T} [GeV] ; efficiency'  jetPt_1_numerator  jetPt_1_denominator",
        "effic_jetEta 'Efficiency vs Jet #eta ; Jet #eta ;           efficiency' jetEta_1_numerator jetEta_1_denominator",
    ),
)

SoftMuHardJetMETEfficiency_metPt = DQMEDHarvester("DQMGenericClient",
    subDirs        = cms.untracked.vstring("HLT/SUSY/SoftMuHardJetMET/MET"),
    verbose        = cms.untracked.uint32(0),
    resolution     = cms.vstring(),
    efficiency     = cms.vstring(
        "effic_metPt 'Efficiency vs MET ; PF MET [GeV] ; efficiency' met_numerator met_denominator",
    ),
)

susyHLTSoftMuHardJetMETClient = cms.Sequence(
    SoftMuHardJetMETEfficiency_muPt
  + SoftMuHardJetMETEfficiency_jetPt
  + SoftMuHardJetMETEfficiency_metPt 
)
