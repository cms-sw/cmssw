import FWCore.ParameterSet.Config as cms

from CMGTools.H2TauTau.objects.cmgTauMu_cfi import cmgTauMu
from CMGTools.H2TauTau.skims.cmgTauMuSel_cfi import cmgTauMuSel

from CMGTools.H2TauTau.objects.cmgTauMuCor_cfi import cmgTauMuCor 
from CMGTools.H2TauTau.objects.tauMuSVFit_cfi import tauMuSVFit 

from CMGTools.H2TauTau.objects.tauCuts_cff import tauPreSelection
from CMGTools.H2TauTau.objects.muCuts_cff import muonPreSelection

from RecoMET.METPUSubtraction.mvaPFMET_cff import pfMVAMEt

# tau pre-selection
tauPreSelectionTauMu = tauPreSelection.clone()
muonPreSelectionTauMu = muonPreSelection.clone()

# mva MET
mvaMETTauMu = cms.EDProducer('PFMETProducerMVATauTau', 
                             **pfMVAMEt.parameters_())#pfMVAMEt.clone()

mvaMETTauMu.srcPFCandidates = cms.InputTag("packedPFCandidates")
mvaMETTauMu.srcVertices = cms.InputTag("offlineSlimmedPrimaryVertices")
mvaMETTauMu.srcLeptons = cms.VInputTag(
  cms.InputTag("tauPreSelectionTauMu", "", ""),
  cms.InputTag("muonPreSelectionTauMu", "", ""),
  )
mvaMETTauMu.permuteLeptons = cms.bool(True)


# Correct tau pt (after MVA MET according to current baseline)
cmgTauMuCor = cmgTauMuCor.clone()

# This selector goes after the tau pt correction
cmgTauMuTauPtSel = cms.EDFilter(
    "PATCompositeCandidateSelector",
    src = cms.InputTag("cmgTauMuCor"),
    cut = cms.string("daughter(0).pt()>18.")
    )

cmgTauMuTauPtSel = cmgTauMuTauPtSel.clone()


# recoil correction
# JAN: We don't know yet if we need this in 2015; re-include if necessary

tauMuMVAMetSequence = cms.Sequence(
    mvaMETTauMu
  )

# SVFit
cmgTauMuCorSVFitPreSel = tauMuSVFit.clone()

# If you want to apply some extra selection after SVFit, do it here
cmgTauMuCorSVFitFullSel = cmgTauMuSel.clone(src = 'cmgTauMuCorSVFitPreSel',
                                            cut = ''
                                            ) 

tauMuSequence = cms.Sequence(   
    tauPreSelectionTauMu +   
    muonPreSelectionTauMu +   
    tauMuMVAMetSequence +
    cmgTauMu +
    cmgTauMuCor+
    cmgTauMuTauPtSel +
    cmgTauMuCorSVFitPreSel +
    cmgTauMuCorSVFitFullSel
  )


