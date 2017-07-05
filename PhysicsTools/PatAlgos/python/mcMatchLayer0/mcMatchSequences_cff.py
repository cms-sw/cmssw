import FWCore.ParameterSet.Config as cms

from PhysicsTools.PatAlgos.mcMatchLayer0.electronMatch_cfi import *
from PhysicsTools.PatAlgos.mcMatchLayer0.muonMatch_cfi import *
from PhysicsTools.PatAlgos.mcMatchLayer0.tauMatch_cfi import *
from PhysicsTools.PatAlgos.mcMatchLayer0.photonMatch_cfi import *
from PhysicsTools.PatAlgos.mcMatchLayer0.ootPhotonMatch_cff import *
from PhysicsTools.PatAlgos.mcMatchLayer0.jetMatch_cfi import *
from PhysicsTools.PatAlgos.mcMatchLayer0.jetFlavourId_cff import *
from PhysicsTools.JetMCAlgos.TauGenJets_cfi import tauGenJets


patMCTruth_LeptonPhoton = cms.Sequence (electronMatch+
                                        muonMatch+
                                        photonMatch+
                                        ootPhotoMatch)

patMCTruth_Jet = cms.Sequence ( patJetPartonMatch +
                                patJetGenJetMatch +
                                patJetFlavourId )

patMCTruth_Tau =  cms.Sequence ( tauMatch+
                                 tauGenJets*
                                 tauGenJetMatch )

patMCTruth = cms.Sequence(patMCTruth_LeptonPhoton+
                          patMCTruth_Jet+
                          patMCTruth_Tau )

