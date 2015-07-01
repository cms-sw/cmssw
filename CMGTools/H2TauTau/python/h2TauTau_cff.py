import FWCore.ParameterSet.Config as cms

from CMGTools.H2TauTau.objects.tauMuObjectsMVAMET_cff import tauMuSequence
from CMGTools.H2TauTau.objects.tauEleObjectsMVAMET_cff import tauEleSequence
from CMGTools.H2TauTau.objects.diTauObjectsMVAMET_cff import diTauSequence
from CMGTools.H2TauTau.objects.diMuObjectsMVAMET_cff import diMuSequence
from CMGTools.H2TauTau.objects.muEleObjectsMVAMET_cff import muEleSequence

from CMGTools.H2TauTau.skims.skim_cff import tauMuFullSelSkimSequence, tauEleFullSelSkimSequence, diTauFullSelSkimSequence, muEleFullSelSkimSequence, diMuFullSelSkimSequence


# Need to explicitly import all modules in all sequences for cms.load(..)
# to work properly in the top-level config

from CMGTools.H2TauTau.objects.mvaMetInputs_cff import mvaMetInputSequence, calibratedAK4PFJetsForPFMVAMEt, puJetIdForPFMVAMEt

from CMGTools.H2TauTau.objects.tauMuObjectsMVAMET_cff import mvaMETTauMu, cmgTauMu, cmgTauMuCor, cmgTauMuTauPtSel, cmgTauMuCorSVFitPreSel, cmgTauMuCorSVFitFullSel, tauMuMVAMetSequence, tauPreSelectionTauMu, muonPreSelectionTauMu

from CMGTools.H2TauTau.objects.tauEleObjectsMVAMET_cff import mvaMETTauEle, cmgTauEle, cmgTauEleCor, cmgTauEleTauPtSel, cmgTauEleCorSVFitPreSel, cmgTauEleCorSVFitFullSel, tauEleMVAMetSequence, tauPreSelectionTauEle, electronPreSelectionTauEle

from CMGTools.H2TauTau.objects.diTauObjectsMVAMET_cff import mvaMETDiTau, cmgDiTau, cmgDiTauCor, cmgDiTauTauPtSel, cmgDiTauCorSVFitPreSel, cmgDiTauCorSVFitFullSel, diTauMVAMetSequence, tauPreSelectionDiTau

from CMGTools.H2TauTau.objects.muEleObjectsMVAMET_cff import mvaMETMuEle, cmgMuEle, cmgMuEleCor, cmgMuEleTauPtSel, cmgMuEleCorSVFitPreSel, cmgMuEleCorSVFitFullSel, muEleMVAMetSequence, muonPreSelectionMuEle, electronPreSelectionMuEle

from CMGTools.H2TauTau.objects.diMuObjectsMVAMET_cff import mvaMETDiMu, cmgDiMu, cmgDiMuCor, cmgDiMuTauPtSel, cmgDiMuCorSVFitPreSel, cmgDiMuCorSVFitFullSel, diMuMVAMetSequence, muonPreSelectionDiMu

from CMGTools.H2TauTau.skims.skim_cff import tauMuFullSelCount, tauEleFullSelCount, diTauFullSelCount, muEleFullSelCount, diMuFullSelCount


# MVA MET Inputs
mvaMetInputPath = cms.Path(
    mvaMetInputSequence
    )

# tau-mu ---
tauMuPath = cms.Path(
    # metRegressionSequence + 
    tauMuSequence + 
    tauMuFullSelSkimSequence
    )

# tau-ele ---
tauElePath = cms.Path(
    # metRegressionSequence + 
    tauEleSequence + 
    tauEleFullSelSkimSequence     
    )

# tau-tau ---
diTauPath = cms.Path(
    # metRegressionSequence + 
    diTauSequence +
    diTauFullSelSkimSequence     
    )

# tau-tau ---
diMuPath = cms.Path(
    # metRegressionSequence + 
    diMuSequence +
    diMuFullSelSkimSequence     
    )

# mu-ele ---
muElePath = cms.Path(
    muEleSequence +
    muEleFullSelSkimSequence     
    )
