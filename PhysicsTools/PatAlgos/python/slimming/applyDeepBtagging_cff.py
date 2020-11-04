import FWCore.ParameterSet.Config as cms

from PhysicsTools.PatAlgos.tools.helpers import getPatAlgosToolsTask, addToProcessAndTask

def applyDeepBtagging( process, postfix="" ) :

    task = getPatAlgosToolsTask(process)

    from PhysicsTools.PatAlgos.tools.jetTools import updateJetCollection

    process.load('PhysicsTools.PatAlgos.slimming.slimmedJets_cfi')
    from Configuration.ProcessModifiers.run2_miniAOD_UL_cff import run2_miniAOD_UL
    from RecoBTag.ONNXRuntime.pfParticleNetAK4_cff import _pfParticleNetAK4JetTagsAll as pfParticleNetAK4JetTagsAll

    # update slimmed jets to include DeepFlavour (keep same name)
    # make clone for DeepFlavour-less slimmed jets, so output name is preserved
    addToProcessAndTask('slimmedJetsNoDeepFlavour', process.slimmedJets.clone(), process, task)
    _btagDiscriminatorsAK4 = cms.PSet( names = cms.vstring(
         'pfDeepFlavourJetTags:probb',
         'pfDeepFlavourJetTags:probbb',
         'pfDeepFlavourJetTags:problepb',
         'pfDeepFlavourJetTags:probc',
         'pfDeepFlavourJetTags:probuds',
         'pfDeepFlavourJetTags:probg',
      )
    )
    run2_miniAOD_UL.toModify(_btagDiscriminatorsAK4,
                             names = _btagDiscriminatorsAK4.names + pfParticleNetAK4JetTagsAll)
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
       btagDiscriminators = _btagDiscriminatorsAK4.names.value(),
       postfix = 'SlimmedDeepFlavour'+postfix,
       printWarning = False
    )

    # slimmedJets with DeepFlavour (remove DeepFlavour-less)
    delattr(process, 'slimmedJets')
    addToProcessAndTask('slimmedJets', getattr(process,'selectedUpdatedPatJetsSlimmedDeepFlavour'+postfix).clone(), process, task)
    # delete module not used anymore (slimmedJets substitutes)
    delattr(process, 'selectedUpdatedPatJetsSlimmedDeepFlavour'+postfix)

    from RecoBTag.ONNXRuntime.pfDeepBoostedJet_cff import _pfDeepBoostedJetTagsAll as pfDeepBoostedJetTagsAll
    from RecoBTag.ONNXRuntime.pfHiggsInteractionNet_cff import _pfHiggsInteractionNetTagsProbs as pfHiggsInteractionNetTagsProbs
    from RecoBTag.ONNXRuntime.pfParticleNet_cff import _pfParticleNetJetTagsAll as pfParticleNetJetTagsAll
    from RecoBTag.ONNXRuntime.pfMassIndependentDeepDoubleXV2JetTags_cff import _pfMassIndependentDeepDoubleXV2JetTagsAll as pfMassIndependentDeepDoubleXV2JetTagsAll

    # update slimmed jets to include particle-based deep taggers (keep same name)
    # make clone for DeepTags-less slimmed AK8 jets, so output name is preserved
    addToProcessAndTask('slimmedJetsAK8NoDeepTags', process.slimmedJetsAK8.clone(), process, task)
    _btagDiscriminators = cms.PSet( names = cms.vstring(
        'pfDeepDoubleBvLJetTags:probQCD',
        'pfDeepDoubleBvLJetTags:probHbb',
        'pfDeepDoubleCvLJetTags:probQCD',
        'pfDeepDoubleCvLJetTags:probHcc',
        'pfDeepDoubleCvBJetTags:probHbb',
        'pfDeepDoubleCvBJetTags:probHcc',
        'pfMassIndependentDeepDoubleBvLJetTags:probQCD',
        'pfMassIndependentDeepDoubleBvLJetTags:probHbb',
        'pfMassIndependentDeepDoubleCvLJetTags:probQCD',
        'pfMassIndependentDeepDoubleCvLJetTags:probHcc',
        'pfMassIndependentDeepDoubleCvBJetTags:probHbb',
        'pfMassIndependentDeepDoubleCvBJetTags:probHcc',
        ) + pfDeepBoostedJetTagsAll
    )
    run2_miniAOD_UL.toModify(_btagDiscriminators,
                             names = _btagDiscriminators.names + pfParticleNetJetTagsAll + pfHiggsInteractionNetTagsProbs + pfMassIndependentDeepDoubleXV2JetTagsAll)
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
       btagDiscriminators = _btagDiscriminators.names.value(),
       postfix = 'SlimmedAK8DeepTags'+postfix,
       printWarning = False
    )

    # slimmedJetsAK8 with DeepTags (remove DeepTags-less)
    delattr(process, 'slimmedJetsAK8')
    addToProcessAndTask('slimmedJetsAK8', getattr(process,'selectedUpdatedPatJetsSlimmedAK8DeepTags'+postfix).clone(), process, task)
    # delete module not used anymore (slimmedJetsAK8 substitutes)
    delattr(process, 'selectedUpdatedPatJetsSlimmedAK8DeepTags'+postfix)


