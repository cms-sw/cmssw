import FWCore.ParameterSet.Config as cms
from PhysicsTools.PatAlgos.tools.puppiJetMETReclusteringTools import puppiAK4METReclusterFromMiniAOD
from PhysicsTools.PatAlgos.tools.puppiJetMETReclusteringTools import puppiAK8ReclusterFromMiniAOD

def setupPuppiAK4AK8METReclustering(process, runOnMC, useExistingWeights=False, reclusterAK4MET=True, reclusterAK8=True, btagDiscriminatorsAK4=None, btagDiscriminatorsAK8=None, btagDiscriminatorsAK8Subjets=None):

  if reclusterAK4MET:
    process = puppiAK4METReclusterFromMiniAOD(process, runOnMC,
      useExistingWeights=useExistingWeights,
      btagDiscriminatorsAK4=btagDiscriminatorsAK4
    )

  if reclusterAK8:
    process = puppiAK8ReclusterFromMiniAOD(process, runOnMC,
      useExistingWeights=useExistingWeights,
      btagDiscriminatorsAK8=btagDiscriminatorsAK8,
      btagDiscriminatorsAK8Subjets=btagDiscriminatorsAK8Subjets
    )

  return process

def puppiJetMETReclusterFromMiniAOD(process, runOnMC, useExistingWeights=False, reclusterAK4MET=True, reclusterAK8=True):

  # AK4 taggers
  from RecoBTag.ONNXRuntime.pfParticleNetFromMiniAODAK4_cff import _pfParticleNetFromMiniAODAK4PuppiCentralJetTagsAll as pfParticleNetFromMiniAODAK4PuppiCentralJetTagsAll
  from RecoBTag.ONNXRuntime.pfParticleNetFromMiniAODAK4_cff import _pfParticleNetFromMiniAODAK4PuppiForwardJetTagsAll as pfParticleNetFromMiniAODAK4PuppiForwardJetTagsAll
  from RecoBTag.ONNXRuntime.pfUnifiedParticleTransformerAK4_cff import _pfUnifiedParticleTransformerAK4JetTagsAll as pfUnifiedParticleTransformerAK4JetTagsAll
  from RecoBTag.ONNXRuntime.pfUnifiedParticleTransformerAK4V1_cff import _pfUnifiedParticleTransformerAK4V1JetTagsAll as pfUnifiedParticleTransformerAK4V1JetTagsAll

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
    + pfUnifiedParticleTransformerAK4V1JetTagsAll
  )

  # AK8 taggers
  from RecoBTag.ONNXRuntime.pfParticleNet_cff import _pfParticleNetJetTagsAll as pfParticleNetJetTagsAll
  from RecoBTag.ONNXRuntime.pfParticleNet_cff import _pfParticleNetMassRegressionOutputs as pfParticleNetMassRegressionOutputs
  from RecoBTag.ONNXRuntime.pfParticleNet_cff import _pfParticleNetMassCorrelatedJetTagsAll as pfParticleNetMassCorrelatedJetTagsAll
  from RecoBTag.ONNXRuntime.pfParticleNetFromMiniAODAK8_cff import _pfParticleNetFromMiniAODAK8JetTagsAll as pfParticleNetFromMiniAODAK8JetTagsAll
  from RecoBTag.ONNXRuntime.pfGlobalParticleTransformerAK8_cff import _pfGlobalParticleTransformerAK8JetTagsAll as pfGlobalParticleTransformerAK8JetTagsAll
  btagDiscriminatorsAK8 = cms.PSet(names = cms.vstring(
      pfParticleNetMassCorrelatedJetTagsAll+
      pfGlobalParticleTransformerAK8JetTagsAll+
      pfParticleNetFromMiniAODAK8JetTagsAll+
      pfParticleNetJetTagsAll+
      pfParticleNetMassRegressionOutputs
    )
  )

  # AK8 Subjets taggers
  btagDiscriminatorsAK8Subjets = cms.PSet(names = cms.vstring(
      'pfDeepFlavourJetTags:probb',
      'pfDeepFlavourJetTags:probbb',
      'pfDeepFlavourJetTags:problepb',
      'pfUnifiedParticleTransformerAK4DiscriminatorsJetTags:BvsAll',
      'pfUnifiedParticleTransformerAK4JetTags:ptcorr',
      'pfUnifiedParticleTransformerAK4JetTags:ptnu',
      'pfUnifiedParticleTransformerAK4JetTags:ptreshigh',
      'pfUnifiedParticleTransformerAK4JetTags:ptreslow',
      'pfUnifiedParticleTransformerAK4V1JetTags:ptcorr',
      'pfUnifiedParticleTransformerAK4V1JetTags:ptnu',
      'pfUnifiedParticleTransformerAK4V1JetTags:ptreshigh',
      'pfUnifiedParticleTransformerAK4V1JetTags:ptreslow',
    )
  )
  process = setupPuppiAK4AK8METReclustering(process, runOnMC,
    useExistingWeights=useExistingWeights,
    reclusterAK4MET=reclusterAK4MET, reclusterAK8=reclusterAK8,
    btagDiscriminatorsAK4=btagDiscriminatorsAK4,
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

