import FWCore.ParameterSet.Config as cms

from CMGTools.H2TauTau.objects.cmgMuEle_cfi import cmgMuEle
from CMGTools.H2TauTau.skims.cmgMuEleSel_cfi import cmgMuEleSel

from CMGTools.H2TauTau.objects.cmgMuEleCor_cfi import cmgMuEleCor 
from CMGTools.H2TauTau.objects.muEleSVFit_cfi import muEleSVFit 

from CMGTools.H2TauTau.objects.muCuts_cff import muonPreSelection
from CMGTools.H2TauTau.objects.eleCuts_cff import electronPreSelection

from RecoMET.METPUSubtraction.mvaPFMET_cff import pfMVAMEt

# lepton pre-selection
muonPreSelectionMuEle = muonPreSelection.clone()
electronPreSelectionMuEle = electronPreSelection.clone()

# mva MET
mvaMETMuEle = cms.EDProducer('PFMETProducerMVATauTau', 
                             **pfMVAMEt.parameters_())

mvaMETMuEle.srcPFCandidates = cms.InputTag("packedPFCandidates")
mvaMETMuEle.srcVertices = cms.InputTag("offlineSlimmedPrimaryVertices")
mvaMETMuEle.srcLeptons = cms.VInputTag(
  cms.InputTag("muonPreSelectionMuEle", "", ""),
  cms.InputTag("electronPreSelectionMuEle", "", ""),
  )
mvaMETMuEle.permuteLeptons = cms.bool(True)



# Correct tau pt (after MVA MET according to current baseline)
cmgMuEleCor = cmgMuEleCor.clone()

# This selector goes after the tau pt correction
cmgMuEleTauPtSel = cms.EDFilter(
    "PATCompositeCandidateSelector",
    src = cms.InputTag("cmgMuEleCor"),
    cut = cms.string("daughter(0).pt()>18.")
    )

cmgMuEleTauPtSel = cmgMuEleTauPtSel.clone()


# recoil correction
# JAN: We don't know yet if we need this in 2015; re-include if necessary

muEleMVAMetSequence = cms.Sequence(
    mvaMETMuEle
  )

# SVFit
cmgMuEleCorSVFitPreSel = muEleSVFit.clone()

# If you want to apply some extra selection after SVFit, do it here
cmgMuEleCorSVFitFullSel = cmgMuEleSel.clone(src = 'cmgMuEleCorSVFitPreSel',
                                              cut = ''
                                              ) 

muEleSequence = cms.Sequence( #
    muonPreSelectionMuEle +   
    electronPreSelectionMuEle +   
    muEleMVAMetSequence +
    cmgMuEle +
    cmgMuEleCor+
    cmgMuEleTauPtSel +
    cmgMuEleCorSVFitPreSel +
    cmgMuEleCorSVFitFullSel
    )
