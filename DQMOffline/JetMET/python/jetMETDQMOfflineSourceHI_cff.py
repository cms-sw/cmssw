import FWCore.ParameterSet.Config as cms

from DQMOffline.JetMET.jetMETAnalyzer_cff import *
from DQMOffline.JetMET.caloTowers_cff     import *
from DQMOffline.JetMET.BeamHaloAnalyzer_cfi import *

AnalyzeBeamHalo.StandardDQM = cms.bool(True)

jetMETAnalyzer.DoPFJetAnalysis  = cms.untracked.bool(False)
jetMETAnalyzer.DoJPTJetAnalysis = cms.untracked.bool(False)
jetMETAnalyzer.DoDiJetSelection = cms.untracked.bool(False)

jetMETAnalyzer.AKJetsCollectionLabel  = cms.InputTag("iterativeConePu5CaloJets")
jetMETAnalyzer.ICJetsCollectionLabel  = cms.InputTag("iterativeConePu5CaloJets")
jetMETAnalyzer.CleaningParameters.vertexLabel = cms.InputTag("hiSelectedVertex")

jetMETAnalyzer.DoPFJetCleaning  = cms.untracked.bool(False)
jetMETAnalyzer.DoJPTJetCleaning = cms.untracked.bool(False)
jetMETAnalyzer.DoJetCleaning    = cms.untracked.bool(True)

jetMETAnalyzer.DoCaloMETAnalysis            = cms.untracked.bool(False)
jetMETAnalyzer.DoTcMETAnalysis              = cms.untracked.bool(False)
jetMETAnalyzer.DoMuCorrMETAnalysis          = cms.untracked.bool(False)
jetMETAnalyzer.DoPfMETAnalysis              = cms.untracked.bool(False)
jetMETAnalyzer.DoHTMHTAnalysis              = cms.untracked.bool(False)

# prevent jetIDHelper from calculating cell based variables which don't make sense w/ HI bkg subtraction 
jetMETAnalyzer.jetAnalysis.JetIDParams.useRecHits = False
jetMETAnalyzer.CleanedjetAnalysis.JetIDParams.useRecHits = False
# cleaned jet folder now uses same jet id parameters as non-cleaned
# cleaning is controlled by PV/DCS filters
jetMETAnalyzer.CleanedjetAnalysis.ptThreshold       = cms.double(20.)
jetMETAnalyzer.CleanedjetAnalysis.n90HitsMin        = cms.int32(-1)
jetMETAnalyzer.CleanedjetAnalysis.fHPDMax           = cms.double(1)
jetMETAnalyzer.CleanedjetAnalysis.resEMFMin         = cms.double(0.)
jetMETAnalyzer.CleanedjetAnalysis.fillJIDPassFrac   = cms.int32(1)

jetMETAnalyzer.CleanedjetAnalysis.fillpfJIDPassFrac = cms.int32(0)
jetMETAnalyzer.CleanedjetAnalysis.ThisCHFMin        = cms.double(-999.)
jetMETAnalyzer.CleanedjetAnalysis.ThisNHFMax        = cms.double(999.)
jetMETAnalyzer.CleanedjetAnalysis.ThisCEFMax        = cms.double(999.)
jetMETAnalyzer.CleanedjetAnalysis.ThisNEFMax        = cms.double(999.)

jetMETAnalyzer.caloMETAnalysis.allHist                = cms.bool(False)

towerSchemeBAnalyzer.AllHist = cms.untracked.bool(False)

jetMETDQMOfflineSource = cms.Sequence(analyzecaloTowersDQM*jetMETAnalyzerSequence)
#jetMETDQMOfflineSource = cms.Sequence(analyzecaloTowersDQM*jetMETAnalyzerSequence)
