import FWCore.ParameterSet.Config as cms

from PhysicsTools.PatAlgos.tools.helpers import getPatAlgosToolsTask, addToProcessAndTask

def applyDeepBtagging( process, postfix="" ) :

    task = getPatAlgosToolsTask(process)

    from PhysicsTools.PatAlgos.tools.jetTools import updateJetCollection

    process.load('PhysicsTools.PatAlgos.slimming.slimmedJets_cfi')

    # update slimmed jets to include DeepFlavour (keep same name)
    # make clone for DeepFlavour-less slimmed jets, so output name is preserved
    addToProcessAndTask('slimmedJetsNoDeepFlavour', process.slimmedJets.clone(), process, task)
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
    addToProcessAndTask('slimmedJets', getattr(process,'selectedUpdatedPatJetsSlimmedDeepFlavour'+postfix).clone(), process, task)
    # delete module not used anymore (slimmedJets substitutes)
    delattr(process, 'selectedUpdatedPatJetsSlimmedDeepFlavour'+postfix)


    # update slimmed jets to include DeepFlavour (keep same name)
    # make clone for DeepDoubleB-less slimmed AK8 jets, so output name is preserved
    addToProcessAndTask('slimmedJetsAK8NoDeepTags', process.slimmedJetsAK8.clone(), process, task)
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
       btagDiscriminators = [
          'pfDeepDoubleBJetTags:probQ',
          'pfDeepDoubleBJetTags:probH',

           # DeepBoostedJet (Nominal)
          'pfDeepBoostedJetTags:probTbcq',
          'pfDeepBoostedJetTags:probTbqq',
          'pfDeepBoostedJetTags:probTbc',
          'pfDeepBoostedJetTags:probTbq',
          'pfDeepBoostedJetTags:probWcq',
          'pfDeepBoostedJetTags:probWqq',
          'pfDeepBoostedJetTags:probZbb',
          'pfDeepBoostedJetTags:probZcc',
          'pfDeepBoostedJetTags:probZqq',
          'pfDeepBoostedJetTags:probHbb',
          'pfDeepBoostedJetTags:probHcc',
          'pfDeepBoostedJetTags:probHqqqq',
          'pfDeepBoostedJetTags:probQCDbb',
          'pfDeepBoostedJetTags:probQCDcc',
          'pfDeepBoostedJetTags:probQCDb',
          'pfDeepBoostedJetTags:probQCDc',
          'pfDeepBoostedJetTags:probQCDothers',
           # meta taggers
          'pfDeepBoostedDiscriminatorsJetTags:TvsQCD',
          'pfDeepBoostedDiscriminatorsJetTags:WvsQCD',

           # DeepBoostedJet (mass decorrelated)
          'pfMassDecorrelatedDeepBoostedJetTags:probTbcq',
          'pfMassDecorrelatedDeepBoostedJetTags:probTbqq',
          'pfMassDecorrelatedDeepBoostedJetTags:probTbc',
          'pfMassDecorrelatedDeepBoostedJetTags:probTbq',
          'pfMassDecorrelatedDeepBoostedJetTags:probWcq',
          'pfMassDecorrelatedDeepBoostedJetTags:probWqq',
          'pfMassDecorrelatedDeepBoostedJetTags:probZbb',
          'pfMassDecorrelatedDeepBoostedJetTags:probZcc',
          'pfMassDecorrelatedDeepBoostedJetTags:probZqq',
          'pfMassDecorrelatedDeepBoostedJetTags:probHbb',
          'pfMassDecorrelatedDeepBoostedJetTags:probHcc',
          'pfMassDecorrelatedDeepBoostedJetTags:probHqqqq',
          'pfMassDecorrelatedDeepBoostedJetTags:probQCDbb',
          'pfMassDecorrelatedDeepBoostedJetTags:probQCDcc',
          'pfMassDecorrelatedDeepBoostedJetTags:probQCDb',
          'pfMassDecorrelatedDeepBoostedJetTags:probQCDc',
          'pfMassDecorrelatedDeepBoostedJetTags:probQCDothers',
           # meta taggers
          'pfMassDecorrelatedDeepBoostedDiscriminatorsJetTags:TvsQCD',
          'pfMassDecorrelatedDeepBoostedDiscriminatorsJetTags:WvsQCD',
          'pfMassDecorrelatedDeepBoostedDiscriminatorsJetTags:bbvsQCD',
          'pfMassDecorrelatedDeepBoostedDiscriminatorsJetTags:ZHbbvsQCD',

       ],
       postfix = 'SlimmedAK8DeepTags'+postfix,
       printWarning = False
    )

    # slimmedJetsAK8 with DeepDoubleB (remove DeepDoubleB-less)
    delattr(process, 'slimmedJetsAK8')
    addToProcessAndTask('slimmedJetsAK8', getattr(process,'selectedUpdatedPatJetsSlimmedAK8DeepTags'+postfix).clone(), process, task)
    # delete module not used anymore (slimmedJetsAK8 substitutes)
    delattr(process, 'selectedUpdatedPatJetsSlimmedAK8DeepTags'+postfix)


