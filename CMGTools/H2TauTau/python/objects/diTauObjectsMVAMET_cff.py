import FWCore.ParameterSet.Config as cms

from CMGTools.H2TauTau.skims.cmgDiTauSel_cfi   import cmgDiTauSel

from CMGTools.H2TauTau.objects.cmgDiTau_cfi    import cmgDiTau
from CMGTools.H2TauTau.objects.cmgDiTauCor_cfi import cmgDiTauCor 
from CMGTools.H2TauTau.objects.diTauSVFit_cfi  import diTauSVFit 
from CMGTools.H2TauTau.objects.tauCuts_cff     import tauPreSelection

from RecoMET.METPUSubtraction.mvaPFMET_cff     import pfMVAMEt

# tau pre-selection
tauPreSelectionDiTau = tauPreSelection.clone(
  #cut = 'pt > 40. && abs(eta) < 2.5 && tauID("decayModeFinding") > 0.5')
  cut = 'pt > 40. && abs(eta) < 2.5 && tauID("decayModeFindingNewDMs") > 0.5') # RIC: new DM. Probably we'd want to save both with an OR and decide later, useful for studying new tauID

# 2012 preselection:
# cut = 'leg1().pt()>40. && leg2().pt()>40. && leg1().tauID("byCombinedIsolationDeltaBetaCorrRaw3Hits") < 10. &&  leg2().tauID("byCombinedIsolationDeltaBetaCorrRaw3Hits") < 10.',

# mva MET

mvaMETDiTau = cms.EDProducer('PFMETProducerMVATauTau', 
                             **pfMVAMEt.parameters_())

mvaMETDiTau.srcPFCandidates = cms.InputTag("packedPFCandidates")
mvaMETDiTau.srcVertices = cms.InputTag("offlineSlimmedPrimaryVertices")
mvaMETDiTau.srcLeptons = cms.VInputTag(
  cms.InputTag("tauPreSelectionDiTau", "", ""),
  cms.InputTag("tauPreSelectionDiTau", "", ""),
  )
mvaMETDiTau.permuteLeptons = cms.bool(True)

 
# correct TauES (after MVA MET according to current baseline)
cmgDiTauCor = cmgDiTauCor.clone()

# this selector goes after the TauES correction
cmgDiTauTauPtSel = cms.EDFilter(
  "PATCompositeCandidateSelector",
  src = cms.InputTag("cmgDiTauCor"),
  cut = cms.string("daughter(0).pt()>45. && daughter(1).pt()>45.")
  )

# recoil correction ----------------------------------------------------
# JAN: We don't know yet if we need this in 2015; re-include if necessary

# sequence -------------------------------------------------------------
diTauMVAMetSequence = cms.Sequence(
  mvaMETDiTau
  )

# SVFit ----------------------------------------------------------------
cmgDiTauCorSVFitPreSel = diTauSVFit.clone()

cmgDiTauCorSVFitFullSel = cmgDiTauSel.clone() 

diTauSequence = cms.Sequence(   
  tauPreSelectionDiTau    +   
  diTauMVAMetSequence     +
  cmgDiTau                +
  cmgDiTauCor             +
  cmgDiTauTauPtSel        +
  cmgDiTauCorSVFitPreSel  +
  cmgDiTauCorSVFitFullSel
  )
