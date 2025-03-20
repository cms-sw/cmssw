import FWCore.ParameterSet.Config as cms

from PhysicsTools.PatAlgos.tools.helpers import getPatAlgosToolsTask, addToProcessAndTask


def applyDeepBtagging(process, postfix=""):
    task = getPatAlgosToolsTask(process)

    from PhysicsTools.PatAlgos.tools.jetTools import updateJetCollection

    from PhysicsTools.PatAlgos.slimming.slimmedJets_cfi import slimmedJets, slimmedJetsAK8
    from RecoBTag.ONNXRuntime.pfParticleNetFromMiniAODAK4_cff import _pfParticleNetFromMiniAODAK4PuppiCentralJetTagsAll as pfParticleNetFromMiniAODAK4PuppiCentralJetTagsAll
    from RecoBTag.ONNXRuntime.pfParticleNetFromMiniAODAK4_cff import _pfParticleNetFromMiniAODAK4PuppiForwardJetTagsAll as pfParticleNetFromMiniAODAK4PuppiForwardJetTagsAll
    from RecoBTag.ONNXRuntime.pfParticleNetFromMiniAODAK4_cff import _pfParticleNetFromMiniAODAK4CHSCentralJetTagsAll as pfParticleNetFromMiniAODAK4CHSCentralJetTagsAll
    from RecoBTag.ONNXRuntime.pfUnifiedParticleTransformerAK4_cff import _pfUnifiedParticleTransformerAK4JetTagsAll as pfUnifiedParticleTransformerAK4JetTagsAll
    from RecoBTag.ONNXRuntime.pfUnifiedParticleTransformerAK4V1_cff import _pfUnifiedParticleTransformerAK4V1JetTagsAll as pfUnifiedParticleTransformerAK4V1JetTagsAll
    
    # update slimmed jets to include DeepFlavour (keep same name)
    # make clone for DeepFlavour-less slimmed jets, so output name is preserved
    addToProcessAndTask('slimmedJetsNoDeepFlavour', slimmedJets.clone(), process, task)
    _btagDiscriminatorsAK4CHS = cms.PSet(
        names=cms.vstring(pfParticleNetFromMiniAODAK4CHSCentralJetTagsAll)
    )
    updateJetCollection(
        process,
        jetSource = cms.InputTag('slimmedJetsNoDeepFlavour'),
        # updateJetCollection defaults to MiniAOD inputs but
        # here it is made explicit (as in training or MINIAOD redoing)
        pvSource = cms.InputTag('offlineSlimmedPrimaryVertices'),
        pfCandidates = cms.InputTag('packedPFCandidates'),
        svSource = cms.InputTag('slimmedSecondaryVertices'),
        muSource = cms.InputTag('slimmedMuons'),
        elSource = cms.InputTag('slimmedElectrons'),
        jetCorrections = ('AK4PFchs', cms.vstring(['L1FastJet', 'L2Relative', 'L3Absolute']), 'None'),
        btagDiscriminators = _btagDiscriminatorsAK4CHS.names.value(),
        postfix = 'SlimmedDeepFlavour' + postfix,
        printWarning = False
    )

    # slimmedJets with DeepFlavour (remove DeepFlavour-less)
    addToProcessAndTask('slimmedJets', getattr(process, 'selectedUpdatedPatJetsSlimmedDeepFlavour' + postfix).clone(), process, task)
    # delete module not used anymore (slimmedJets substitutes)
    delattr(process, 'selectedUpdatedPatJetsSlimmedDeepFlavour' + postfix)

    # update slimmedJetsPuppi to include deep taggers
    addToProcessAndTask('slimmedJetsPuppiNoDeepTags', slimmedJets.clone(
        src = "selectedPatJetsPuppi", packedPFCandidates = "packedPFCandidates"
    ), process, task)
    _btagDiscriminatorsAK4Puppi = cms.PSet(
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
  
    updateJetCollection(
        process,
        jetSource = cms.InputTag('slimmedJetsPuppiNoDeepTags'),
        pvSource = cms.InputTag('offlineSlimmedPrimaryVertices'),
        pfCandidates = cms.InputTag('packedPFCandidates'),
        svSource = cms.InputTag('slimmedSecondaryVertices'),
        muSource = cms.InputTag('slimmedMuons'),
        elSource = cms.InputTag('slimmedElectrons'),
        jetCorrections = ('AK4PFPuppi', cms.vstring(['L2Relative', 'L3Absolute']), 'None'),
        btagDiscriminators = _btagDiscriminatorsAK4Puppi.names.value(),
        postfix = 'SlimmedPuppiWithDeepTags' + postfix,
        printWarning = False
    )

    addToProcessAndTask('slimmedJetsPuppi', getattr(process, 'selectedUpdatedPatJetsSlimmedPuppiWithDeepTags' + postfix).clone(), process, task)
    # delete module not used anymore (slimmedJetsPuppi substitutes)
    delattr(process, 'selectedUpdatedPatJetsSlimmedPuppiWithDeepTags' + postfix)

    from RecoBTag.ONNXRuntime.pfParticleNet_cff import _pfParticleNetJetTagsAll as pfParticleNetJetTagsAll
    from RecoBTag.ONNXRuntime.pfParticleNet_cff import _pfParticleNetMassRegressionOutputs as pfParticleNetMassRegressionOutputs
    from RecoBTag.ONNXRuntime.pfParticleNet_cff import _pfParticleNetMassCorrelatedJetTagsAll as pfParticleNetMassCorrelatedJetTagsAll
    from RecoBTag.ONNXRuntime.pfParticleNetFromMiniAODAK8_cff import _pfParticleNetFromMiniAODAK8JetTagsAll as pfParticleNetFromMiniAODAK8JetTagsAll
    from RecoBTag.ONNXRuntime.pfGlobalParticleTransformerAK8_cff import _pfGlobalParticleTransformerAK8JetTagsAll as pfGlobalParticleTransformerAK8JetTagsAll

    # update slimmed jets to include particle-based deep taggers (keep same name)
    # make clone for DeepTags-less slimmed AK8 jets, so output name is preserved
    addToProcessAndTask('slimmedJetsAK8NoDeepTags', slimmedJetsAK8.clone(), process, task)
    _btagDiscriminatorsAK8 = cms.PSet(names = cms.vstring(pfParticleNetMassCorrelatedJetTagsAll+pfGlobalParticleTransformerAK8JetTagsAll+pfParticleNetFromMiniAODAK8JetTagsAll+pfParticleNetJetTagsAll+pfParticleNetMassRegressionOutputs))
    updateJetCollection(
        process,
        jetSource = cms.InputTag('slimmedJetsAK8NoDeepTags'),
        # updateJetCollection defaults to MiniAOD inputs but
        # here it is made explicit (as in training or MINIAOD redoing)
        pvSource = cms.InputTag('offlineSlimmedPrimaryVertices'),
        pfCandidates = cms.InputTag('packedPFCandidates'),
        svSource = cms.InputTag('slimmedSecondaryVertices'),
        muSource = cms.InputTag('slimmedMuons'),
        elSource = cms.InputTag('slimmedElectrons'),
        rParam = 0.8,
        jetCorrections = ('AK8PFPuppi', cms.vstring(['L2Relative', 'L3Absolute']), 'None'),
        btagDiscriminators = _btagDiscriminatorsAK8.names.value(),
        postfix = 'SlimmedAK8DeepTags' + postfix,
        printWarning = False
    )

    # slimmedJetsAK8 with DeepTags (remove DeepTags-less)
    addToProcessAndTask('slimmedJetsAK8', getattr(process, 'selectedUpdatedPatJetsSlimmedAK8DeepTags' + postfix).clone(), process, task)
    # delete module not used anymore (slimmedJetsAK8 substitutes)
    delattr(process, 'selectedUpdatedPatJetsSlimmedAK8DeepTags' + postfix)
