import FWCore.ParameterSet.Config as cms

from HLTrigger.btau.jetTag.lifetimeHLT_cff import *
from HLTrigger.btau.jetTag.softmuonHLT_cff import *
from HLTrigger.btau.displacedmumu.PathBToJpsiToMumuLoose_cff import *
from HLTrigger.btau.displacedmumu.HLTmmkFilter_cff import *
from HLTrigger.btau.tau.TauHLT_cff import *
HLTB1Jet = cms.Path(hltBLifetime1jet+cms.SequencePlaceholder("hltEnd"))
HLTB2Jet = cms.Path(hltBLifetime2jet+cms.SequencePlaceholder("hltEnd"))
HLTB3Jet = cms.Path(hltBLifetime3jet+cms.SequencePlaceholder("hltEnd"))
HLTB4Jet = cms.Path(hltBLifetime4jet+cms.SequencePlaceholder("hltEnd"))
HLTBHT = cms.Path(hltBLifetimeHT+cms.SequencePlaceholder("hltEnd"))
HLTB1JetMu = cms.Path(hltBSoftmuon1jet+cms.SequencePlaceholder("hltEnd"))
HLTB2JetMu = cms.Path(hltBSoftmuon2jet+cms.SequencePlaceholder("hltEnd"))
HLTB3JetMu = cms.Path(hltBSoftmuon3jet+cms.SequencePlaceholder("hltEnd"))
HLTB4JetMu = cms.Path(hltBSoftmuon4jet+cms.SequencePlaceholder("hltEnd"))
HLTBHTMu = cms.Path(hltBSoftmuonHT+cms.SequencePlaceholder("hltEnd"))
HLTBJPsiMuMu = cms.Path(btoJpsitoMumu+cms.SequencePlaceholder("hltEnd"))
CandHLTBToMuMuK = cms.Path(BToMuMuK+cms.SequencePlaceholder("hltEnd"))
HLT1Tau = cms.Path(singleTauL1HLT+cms.SequencePlaceholder("hltEnd"))
HLT1Tau1MET = cms.Path(singleTauMETL1HLT+cms.SequencePlaceholder("hltEnd"))
HLT2TauPixel = cms.Path(caloPixelTauL1HLT+cms.SequencePlaceholder("hltEnd"))

