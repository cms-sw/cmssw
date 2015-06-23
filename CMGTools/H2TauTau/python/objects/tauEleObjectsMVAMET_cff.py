import FWCore.ParameterSet.Config as cms

from CMGTools.H2TauTau.objects.cmgTauEle_cfi import cmgTauEle
from CMGTools.H2TauTau.skims.cmgTauEleSel_cfi import cmgTauEleSel

from CMGTools.H2TauTau.objects.cmgTauEleCor_cfi import cmgTauEleCor 
from CMGTools.H2TauTau.objects.tauEleSVFit_cfi import tauEleSVFit 

from CMGTools.H2TauTau.objects.tauCuts_cff import tauPreSelection
from CMGTools.H2TauTau.objects.eleCuts_cff import electronPreSelection

from RecoMET.METPUSubtraction.mvaPFMET_cff import pfMVAMEt

# tau pre-selection
tauPreSelectionTauEle = tauPreSelection.clone()
electronPreSelectionTauEle = electronPreSelection.clone()

# mva MET
mvaMETTauEle = cms.EDProducer('PFMETProducerMVATauTau', 
                             **pfMVAMEt.parameters_())

mvaMETTauEle.srcPFCandidates = cms.InputTag("packedPFCandidates")
mvaMETTauEle.srcVertices = cms.InputTag("offlineSlimmedPrimaryVertices")
mvaMETTauEle.srcLeptons = cms.VInputTag(
  cms.InputTag("tauPreSelectionTauEle", "", ""),
  cms.InputTag("electronPreSelectionTauEle", "", ""),
  )
mvaMETTauEle.permuteLeptons = cms.bool(True)



# Correct tau pt (after MVA MET according to current baseline)
cmgTauEleCor = cmgTauEleCor.clone()

# This selector goes after the tau pt correction
cmgTauEleTauPtSel = cms.EDFilter(
    "PATCompositeCandidateSelector",
    src = cms.InputTag("cmgTauEleCor"),
    cut = cms.string("daughter(0).pt()>18.")
    )

cmgTauEleTauPtSel = cmgTauEleTauPtSel.clone()


# recoil correction
# JAN: We don't know yet if we need this in 2015; re-include if necessary

tauEleMVAMetSequence = cms.Sequence(
    mvaMETTauEle
  )

# SVFit
cmgTauEleCorSVFitPreSel = tauEleSVFit.clone()

# If you want to apply some extra selection after SVFit, do it here
cmgTauEleCorSVFitFullSel = cmgTauEleSel.clone(src = 'cmgTauEleCorSVFitPreSel',
                                              cut = ''
                                              ) 

tauEleSequence = cms.Sequence( #
    tauPreSelectionTauEle +   
    electronPreSelectionTauEle +   
    tauEleMVAMetSequence +
    cmgTauEle +
    cmgTauEleCor+
    cmgTauEleTauPtSel +
    cmgTauEleCorSVFitPreSel +
    cmgTauEleCorSVFitFullSel
    )
