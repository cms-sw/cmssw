import FWCore.ParameterSet.Config as cms

from PhysicsTools.PatAlgos.tools.helpers import getPatAlgosToolsTask, addToProcessAndTask

def applyDeepBtagging( process, postfix="" ) :

    task = getPatAlgosToolsTask(process)

    process.load('PhysicsTools.PatAlgos.slimming.slimmedJets_cfi')

    # update slimmed jets to include DeepFlavour (keep same name)
    from PhysicsTools.PatAlgos.tools.jetTools import updateJetCollection
    # make clone for DeepFlavour-less slimmed jets, so output name is preserved
    process.slimmedJetsNoDeepFlavour = process.slimmedJets.clone()
    task.add(process.slimmedJetsNoDeepFlavour)
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
       btagDiscriminators = [
          'pfDeepFlavourJetTags:probb',
          'pfDeepFlavourJetTags:probbb',
          'pfDeepFlavourJetTags:problepb',
          'pfDeepFlavourJetTags:probc',
          'pfDeepFlavourJetTags:probuds',
          'pfDeepFlavourJetTags:probg',
       ],
       postfix = 'SlimmedDeepFlavour'+postfix,
       printWarning = False
    )

    # slimmedJets with DeepFlavour (remove DeepFlavour-less)
    delattr(process, 'slimmedJets')
    process.slimmedJets = process.selectedUpdatedPatJetsSlimmedDeepFlavour.clone()
    # delete module not used anymore (slimmedJets substitutes)
    delattr(process, 'selectedUpdatedPatJetsSlimmedDeepFlavour'+postfix)

    task.add(process.slimmedJets)

    # update slimmed jets to include DeepFlavour (keep same name)
    # make clone for DeepDoubleB-less slimmed AK8 jets, so output name is preserved
    process.slimmedJetsAK8NoDeepDoubleB = process.slimmedJetsAK8.clone()
    task.add(process.slimmedJetsAK8NoDeepDoubleB)
    updateJetCollection(
       process,
       jetSource = cms.InputTag('slimmedJetsAK8NoDeepDoubleB'),
       # updateJetCollection defaults to MiniAOD inputs but
       # here it is made explicit (as in training or MINIAOD redoing)
       pvSource = cms.InputTag('offlineSlimmedPrimaryVertices'),
       pfCandidates = cms.InputTag('packedPFCandidates'),
       svSource = cms.InputTag('slimmedSecondaryVertices'),
       muSource = cms.InputTag('slimmedMuons'),
       elSource = cms.InputTag('slimmedElectrons'),
       rParam = 0.8,
       jetCorrections = ('AK8PFchs', cms.vstring(['L1FastJet', 'L2Relative', 'L3Absolute']), 'None'),
       btagDiscriminators = [
          'pfDeepDoubleBJetTags:probQ',
          'pfDeepDoubleBJetTags:probH',
       ],
       postfix = 'SlimmedAK8DeepDoubleB'+postfix,
       printWarning = False
    )

    # slimmedJetsAK8 with DeepDoubleB (remove DeepDoubleB-less)
    delattr(process, 'slimmedJetsAK8')
    process.slimmedJetsAK8 = process.selectedUpdatedPatJetsSlimmedAK8DeepDoubleB.clone()
    # delete module not used anymore (slimmedJetsAK8 substitutes)
    delattr(process, 'selectedUpdatedPatJetsSlimmedAK8DeepDoubleB'+postfix)

    task.add(process.slimmedJetsAK8)

