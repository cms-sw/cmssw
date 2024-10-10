import FWCore.ParameterSet.Config as cms
from PhysicsTools.PatAlgos.tools.puppiJetMETReclusteringTools import puppiAK4METReclusterFromMiniAOD
from PhysicsTools.PatAlgos.tools.puppiJetMETReclusteringTools import puppiAK8ReclusterFromMiniAOD

def puppiJetMETReclusterFromMiniAOD(process, runOnMC, useExistingWeights=False, doAK4MET=True, doAK8=True):

  #
  # AK4 and MET
  #
  from RecoBTag.ONNXRuntime.pfParticleNetFromMiniAODAK4_cff import _pfParticleNetFromMiniAODAK4PuppiCentralJetTagsAll as pfParticleNetFromMiniAODAK4PuppiCentralJetTagsAll
  from RecoBTag.ONNXRuntime.pfParticleNetFromMiniAODAK4_cff import _pfParticleNetFromMiniAODAK4PuppiForwardJetTagsAll as pfParticleNetFromMiniAODAK4PuppiForwardJetTagsAll
  from RecoBTag.ONNXRuntime.pfUnifiedParticleTransformerAK4_cff import _pfUnifiedParticleTransformerAK4JetTagsAll as pfUnifiedParticleTransformerAK4JetTagsAll

  btagDiscriminatorsAK4 = cms.PSet(
   names=cms.vstring(
    'pfDeepFlavourJetTags:probb',
    'pfDeepFlavourJetTags:probbb',
    'pfDeepFlavourJetTags:problepb',
    'pfDeepFlavourJetTags:probc',
    'pfDeepFlavourJetTags:probuds',
    'pfDeepFlavourJetTags:probg')
    + pfParticleNetFromMiniAODAK4PuppiCentralJetTagsAll
    + pfParticleNetFromMiniAODAK4PuppiForwardJetTagsAll
    + pfUnifiedParticleTransformerAK4JetTagsAll
  )

  if doAK4MET:
    process = puppiAK4METReclusterFromMiniAOD(process, runOnMC,
      useExistingWeights=useExistingWeights,
      btagDiscriminatorsAK4=btagDiscriminatorsAK4
    )

  #
  # AK8
  #
  from RecoBTag.ONNXRuntime.pfParticleNet_cff import _pfParticleNetJetTagsAll as pfParticleNetJetTagsAll
  from RecoBTag.ONNXRuntime.pfParticleNet_cff import _pfParticleNetMassRegressionOutputs as pfParticleNetMassRegressionOutputs
  from RecoBTag.ONNXRuntime.pfParticleNet_cff import _pfParticleNetMassCorrelatedJetTagsAll as pfParticleNetMassCorrelatedJetTagsAll
  from RecoBTag.ONNXRuntime.pfParticleNetFromMiniAODAK8_cff import _pfParticleNetFromMiniAODAK8JetTagsAll as pfParticleNetFromMiniAODAK8JetTagsAll

  btagDiscriminatorsAK8 = cms.PSet(names = cms.vstring(
      pfParticleNetMassCorrelatedJetTagsAll+
      pfParticleNetFromMiniAODAK8JetTagsAll+
      pfParticleNetJetTagsAll+
      pfParticleNetMassRegressionOutputs
    )
  )

  btagDiscriminatorsAK8Subjets = cms.PSet(names = cms.vstring(
      'pfDeepFlavourJetTags:probb',
      'pfDeepFlavourJetTags:probbb',
      'pfDeepFlavourJetTags:problepb',
      'pfUnifiedParticleTransformerAK4DiscriminatorsJetTags:BvsAll'
    )
  )

  if doAK8:
    process = puppiAK8ReclusterFromMiniAOD(process, runOnMC,
      useExistingWeights=useExistingWeights,
      btagDiscriminatorsAK8=btagDiscriminatorsAK8,
      btagDiscriminatorsAK8Subjets=btagDiscriminatorsAK8Subjets
    )

  return process

def puppiJetMETReclusterFromMiniAOD_MC(process):
  process = puppiJetMETReclusterFromMiniAOD(process, runOnMC=True)
  return process

def puppiJetMETReclusterFromMiniAOD_Data(process):
  process = puppiJetMETReclusterFromMiniAOD(process, runOnMC=False)
  return process

