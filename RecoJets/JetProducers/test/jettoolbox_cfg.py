## import skeleton process
from PhysicsTools.PatAlgos.patTemplate_cfg import *
## switch to uncheduled mode
process.options.allowUnscheduled = cms.untracked.bool(True)
#process.Tracer = cms.Service('Tracer')

process.load('PhysicsTools.PatAlgos.producersLayer1.patCandidates_cff')
process.load('PhysicsTools.PatAlgos.selectionLayer1.selectedPatCandidates_cff')
process.load("RecoJets.Configuration.RecoGenJets_cff")
process.load("RecoJets.Configuration.GenJetParticles_cff")

process.ca8GenJetsNoNu = process.ca6GenJetsNoNu.clone( rParam = 0.8 )

from PhysicsTools.PatAlgos.tools.jetTools import addJetCollection
from PhysicsTools.PatAlgos.tools.jetTools import switchJetCollection

addJetCollection(
    process,
    labelName = 'AK5PFCHS',
    jetSource = cms.InputTag('ak5PFJetsCHS'),
    algo = 'ak5',
    jetCorrections = ('AK5PFchs', cms.vstring(['L1FastJet', 'L2Relative', 'L3Absolute']), 'None')
    )

addJetCollection(
    process,
    labelName = 'CA8PFCHS',
    jetSource = cms.InputTag('ca8PFJetsCHS'),
    algo = 'ca8',
    jetCorrections = ('AK7PFchs', cms.vstring(['L1FastJet', 'L2Relative', 'L3Absolute']), 'None')
    )

addJetCollection(
    process,
    labelName = 'AK8PFCHS',
    jetSource = cms.InputTag('ak8PFJetsCHS'),
    algo = 'ak8',
    jetCorrections = ('AK7PFchs', cms.vstring(['L1FastJet', 'L2Relative', 'L3Absolute']), 'None')
    )

switchJetCollection(
    process,
    jetSource = cms.InputTag('ak5PFJets'),
    jetCorrections = ('AK5PF', cms.vstring(['L1FastJet', 'L2Relative', 'L3Absolute']), 'Type-1'),
    btagDiscriminators = ['jetBProbabilityBJetTags',
                          'jetProbabilityBJetTags',
                          'trackCountingHighPurBJetTags',
                          'trackCountingHighEffBJetTags',
                          'simpleSecondaryVertexHighEffBJetTags',
                          'simpleSecondaryVertexHighPurBJetTags',
                          'combinedSecondaryVertexBJetTags'
                          ],
    )

process.patJetGenJetMatchPatJetsCA8PFCHS.matched = cms.InputTag("ca8GenJetsNoNu")

patJetsAK5 = process.patJetsAK5PFCHS
patJetsCA8 = process.patJetsCA8PFCHS
patJetsAK8 = process.patJetsAK8PFCHS

process.out.outputCommands += ['keep *_ak5PFJetsCHS_*_*',
                               'keep *_patJetsAK5PFCHS_*_*',
                               'keep *_ca8PFJetsCHS_*_*',
                               'keep *_patJetsCA8PFCHS_*_*',
                               'keep *_ak8PFJetsCHS_*_*',
                               'keep *_patJetsAK8PFCHS_*_*']

####################################################################################################
#THE JET TOOLBOX

#load the various tools

#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#pileupJetID

process.load('RecoJets.JetProducers.pileupjetidproducer_cfi')

process.pileupJetIdCalculator.jets = cms.InputTag("ak5PFJetsCHS")
process.pileupJetIdEvaluator.jets = cms.InputTag("ak5PFJetsCHS")
process.pileupJetIdCalculator.rho = cms.InputTag("fixedGridRhoFastjetAll")
process.pileupJetIdEvaluator.rho = cms.InputTag("fixedGridRhoFastjetAll")

patJetsAK5.userData.userFloats.src += ['pileupJetIdEvaluator:fullDiscriminant']
patJetsAK5.userData.userInts.src += ['pileupJetIdEvaluator:cutbasedId','pileupJetIdEvaluator:fullId']
process.out.outputCommands += ['keep *_pileupJetIdEvaluator_*_*']

#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#QGTagger
"""
process.load('RecoJets.JetProducers.QGTagger_cfi')

process.QGTagger.srcJets = cms.InputTag("ak5PFJetsCHS")

patJetsAK5.userData.userFloats.src += ['QGTagger:qgLikelihood']
process.out.outputCommands += ['keep *_QGTagger_*_*']
"""
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#Njettiness

process.load('RecoJets.JetProducers.nJettinessAdder_cfi')

process.NjettinessCA8 = process.Njettiness.clone()
process.NjettinessCA8.src = cms.InputTag("ca8PFJetsCHS")
process.NjettinessCA8.cone = cms.double(0.8)

patJetsCA8.userData.userFloats.src += ['NjettinessCA8:tau1','NjettinessCA8:tau2','NjettinessCA8:tau3']
process.out.outputCommands += ['keep *_NjettinessCA8_*_*']

process.NjettinessAK8 = process.NjettinessCA8.clone()
process.NjettinessAK8.src = cms.InputTag("ak8PFJetsCHS")

patJetsAK8.userData.userFloats.src += ['NjettinessAK8:tau1','NjettinessAK8:tau2','NjettinessAK8:tau3']
process.out.outputCommands += ['keep *_NjettinessAK8_*_*']

#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#QJetsAdder

process.RandomNumberGeneratorService = cms.Service("RandomNumberGeneratorService",
                                                   QJetsAdderCA8 = cms.PSet(initialSeed = cms.untracked.uint32(7)),
                                                   QJetsAdderAK8 = cms.PSet(initialSeed = cms.untracked.uint32(31)))

process.load('RecoJets.JetProducers.qjetsadder_cfi')

process.QJetsAdderCA8 = process.QJetsAdder.clone()
process.QJetsAdderCA8.src = cms.InputTag("ca8PFJetsCHS")
process.QJetsAdderCA8.jetRad = cms.double(0.8)
process.QJetsAdderCA8.jetAlgo = cms.string('CA')

patJetsCA8.userData.userFloats.src += ['QJetsAdderCA8:QjetsVolatility']
process.out.outputCommands += ['keep *_QJetsAdderCA8_*_*']

process.QJetsAdderAK8 = process.QJetsAdderCA8.clone()
process.QJetsAdderAK8.src = cms.InputTag("ak8PFJetsCHS")
process.QJetsAdderAK8.jetAlgo = cms.string('AK')

patJetsAK8.userData.userFloats.src += ['QJetsAdderAK8:QjetsVolatility']
process.out.outputCommands += ['keep *_QJetsAdderAK8_*_*']
                                   
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#Grooming valueMaps

process.load('RecoJets.Configuration.RecoPFJets_cff')

patJetsCA8.userData.userFloats.src += ['ca8PFJetsCHSPrunedLinks','ca8PFJetsCHSTrimmedLinks','ca8PFJetsCHSFilteredLinks']
process.out.outputCommands += ['keep *_ca8PFJetsCHSPrunedLinks_*_*',
                               'keep *_ca8PFJetsCHSTrimmedLinks_*_*',
                               'keep *_ca8PFJetsCHSFilteredLinks_*_*']

patJetsAK8.userData.userFloats.src += ['ak8PFJetsCHSPrunedLinks','ak8PFJetsCHSTrimmedLinks','ak8PFJetsCHSFilteredLinks']
process.out.outputCommands += ['keep *_ak8PFJetsCHSPrunedLinks_*_*',
                               'keep *_ak8PFJetsCHSTrimmedLinks_*_*',
                               'keep *_ak8PFJetsCHSFilteredLinks_*_*']

process.cmsTopTagPFJetsCHSLinksCA8 = process.ca8PFJetsCHSPrunedLinks.clone()
process.cmsTopTagPFJetsCHSLinksCA8.src = cms.InputTag("ca8PFJetsCHS")
process.cmsTopTagPFJetsCHSLinksCA8.matched = cms.InputTag("cmsTopTagPFJetsCHS")

patJetsCA8.userData.userFloats.src += ['cmsTopTagPFJetsCHSLinksCA8']
process.out.outputCommands += ['keep *_cmsTopTagPFJetsCHSLinksCA8_*_*']

process.cmsTopTagPFJetsCHSLinksAK8 = process.cmsTopTagPFJetsCHSLinksCA8.clone()
process.cmsTopTagPFJetsCHSLinksAK8.src = cms.InputTag("ak8PFJetsCHS")
process.cmsTopTagPFJetsCHSLinksAK8.matched = cms.InputTag("cmsTopTagPFJetsCHS")

patJetsAK8.userData.userFloats.src += ['cmsTopTagPFJetsCHSLinksAK8']
process.out.outputCommands += ['keep *_cmsTopTagPFJetsCHSLinksAK8_*_*']

####################################################################################################

## ------------------------------------------------------
#  In addition you usually want to change the following
#  parameters:
## ------------------------------------------------------
#
#   process.GlobalTag.globaltag =  ...    ##  (according to https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideFrontierConditions)
#                                         ##
from PhysicsTools.PatAlgos.patInputFiles_cff import filesRelValProdTTbarAODSIM
process.source.fileNames = filesRelValProdTTbarAODSIM
#                                         ##
process.maxEvents.input = 5
#                                         ##
#   process.out.outputCommands = [ ... ]  ##  (e.g. taken from PhysicsTools/PatAlgos/python/patEventContent_cff.py)
#                                         ##
process.out.fileName = 'jettoolbox.root'
#                                         ##
#   process.options.wantSummary = False   ##  (to suppress the long output at the end of the job)
