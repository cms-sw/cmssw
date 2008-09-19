import FWCore.ParameterSet.Config as cms

from PhysicsTools.PatAlgos.mcMatchLayer0.electronMatch_cfi import *
from PhysicsTools.PatAlgos.mcMatchLayer0.muonMatch_cfi import *
from PhysicsTools.PatAlgos.mcMatchLayer0.tauMatch_cfi import *
from PhysicsTools.PatAlgos.mcMatchLayer0.photonMatch_cfi import *
from PhysicsTools.PatAlgos.mcMatchLayer0.jetMatch_cfi import *
from PhysicsTools.JetMCAlgos.TauGenJets_cfi import tauGenJets


patMCTruth_LeptonPhoton = cms.Sequence (electronMatch+
                                        muonMatch+
                                        photonMatch)

patMCTruth_Jet = cms.Sequence (jetPartonMatch+
                               jetGenJetMatch)

patMCTruth_Tau =  cms.Sequence ( tauMatch+
                                 tauGenJets*
                                 tauGenJetMatch )

patMCTruth_withoutTau = cms.Sequence(patMCTruth_LeptonPhoton+
                                     patMCTruth_Jet)

patMCTruth_withoutLeptonPhoton = cms.Sequence(patMCTruth_Jet+
                                              patMCTruth_Tau )

patMCTruth = cms.Sequence(patMCTruth_LeptonPhoton+
                          patMCTruth_Jet+
                          patMCTruth_Tau )

