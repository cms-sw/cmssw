import FWCore.ParameterSet.Config as cms

from HLTrigger.xchannel.PathElectronB_local_cff import *
from HLTrigger.xchannel.PathMuB_cff import *
from HLTrigger.xchannel.PathMuBsoftMu_cff import *
from HLTrigger.xchannel.PathElectronPlusJet_cff import *
from HLTrigger.xchannel.PathMuJets_cff import *
from HLTrigger.xchannel.PathMuNoL2IsoJets_cff import *
from HLTrigger.xchannel.PathMuNoIsoJets_cff import *
from HLTrigger.xchannel.electronmuon_cff import *
from HLTrigger.xchannel.electronmuonNonIsolated_cff import *
from HLTrigger.xchannel.LeptonTauHLT_cff import *
HLTXElectronBJet = cms.Path(ElectronB+cms.SequencePlaceholder("hltEnd"))
HLTXMuonBJet = cms.Path(MuB+cms.SequencePlaceholder("hltEnd"))
HLTXMuonBJetSoftMuon = cms.Path(MuBsmu+cms.SequencePlaceholder("hltEnd"))
HLTXElectron1Jet = cms.Path(HLTEplus1jet+cms.SequencePlaceholder("hltEnd"))
HLTXElectron2Jet = cms.Path(HLTEplus2jet+cms.SequencePlaceholder("hltEnd"))
HLTXElectron3Jet = cms.Path(HLTEplus3jet+cms.SequencePlaceholder("hltEnd"))
HLTXElectron4Jet = cms.Path(HLTEplus4jet+cms.SequencePlaceholder("hltEnd"))
HLTXMuonJets = cms.Path(HLTMuIsoJet+cms.SequencePlaceholder("hltEnd"))
CandHLTXMuonNoL2IsoJets = cms.Path(HLTMuNoL2IsoJet+cms.SequencePlaceholder("hltEnd"))
CandHLTXMuonNoIsoJets = cms.Path(HLTMuNoIsoJet+cms.SequencePlaceholder("hltEnd"))
HLTXElectronMuon = cms.Path(emuonsequence+cms.SequencePlaceholder("hltEnd"))
HLTXElectronMuonRelaxed = cms.Path(NonIsoElectronMuonSequence+cms.SequencePlaceholder("hltEnd"))
HLTXElectronTau = cms.Path(hltElectronTau+cms.SequencePlaceholder("hltEnd"))
CandHLTXElectronTauPixel = cms.Path(hltElectronTauPixel+cms.SequencePlaceholder("hltEnd"))
HLTXMuonTau = cms.Path(muonPixelTauL1HLT+cms.SequencePlaceholder("hltEnd"))

