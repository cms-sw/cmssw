## import skeleton process
from PhysicsTools.PatAlgos.patTemplate_cfg import *
## switch to uncheduled mode
process.options.allowUnscheduled = cms.untracked.bool(True)
#process.Tracer = cms.Service('Tracer')

process.load('PhysicsTools.PatAlgos.producersLayer1.patCandidates_cff')
process.load('PhysicsTools.PatAlgos.selectionLayer1.selectedPatCandidates_cff')
process.load("RecoJets.Configuration.RecoGenJets_cff")
process.load("RecoJets.Configuration.GenJetParticles_cff")

#process.ca8GenJetsNoNu = process.ca6GenJetsNoNu.clone( rParam = 0.8 )

from PhysicsTools.PatAlgos.tools.jetTools import addJetCollection
from PhysicsTools.PatAlgos.tools.jetTools import switchJetCollection

addJetCollection(
    process,
    labelName = 'AK4PFCHS',
    jetSource = cms.InputTag('ak4PFJetsCHS'),
    algo = 'ak4',
    rParam = 0.4,
    jetCorrections = ('AK5PFchs', cms.vstring(['L1FastJet', 'L2Relative', 'L3Absolute']), 'None')
    )

addJetCollection(
    process,
    labelName = 'CA8PFCHS',
    jetSource = cms.InputTag('ca8PFJetsCHS'),
    algo = 'ca8',
    rParam = 0.8,
    jetCorrections = ('AK7PFchs', cms.vstring(['L1FastJet', 'L2Relative', 'L3Absolute']), 'None')
    )

addJetCollection(
    process,
    labelName = 'AK8PFCHS',
    jetSource = cms.InputTag('ak8PFJetsCHS'),
    algo = 'ak8',
    rParam = 0.8,
    jetCorrections = ('AK7PFchs', cms.vstring(['L1FastJet', 'L2Relative', 'L3Absolute']), 'None')
    )

switchJetCollection(
    process,
    jetSource = cms.InputTag('ak4PFJets'),
    rParam = 0.4,
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

#process.patJetGenJetMatchCA8PFCHS.matched = cms.InputTag("ca8GenJetsNoNu")

patJetsAK4 = process.patJetsAK4PFCHS
patJetsCA8 = process.patJetsCA8PFCHS
patJetsAK8 = process.patJetsAK8PFCHS

process.out.outputCommands += ['keep *_ak4PFJetsCHS_*_*',
                               'keep *_patJetsAK4PFCHS_*_*',
                               'keep *_ca8PFJetsCHS_*_*',
                               'keep *_patJetsCA8PFCHS_*_*',
                               'keep *_ak8PFJetsCHS_*_*',
                               'keep *_patJetsAK8PFCHS_*_*']

####################################################################################################
#THE JET TOOLBOX

#load the various tools

#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#pileupJetID

process.load('RecoJets.JetProducers.PileupJetID_cfi')

process.pileupJetIdCalculator.jets = cms.InputTag("ak4PFJetsCHS")
process.pileupJetIdEvaluator.jets = cms.InputTag("ak4PFJetsCHS")
process.pileupJetIdCalculator.rho = cms.InputTag("fixedGridRhoFastjetAll")
process.pileupJetIdEvaluator.rho = cms.InputTag("fixedGridRhoFastjetAll")

patJetsAK4.userData.userFloats.src += ['pileupJetIdEvaluator:fullDiscriminant']
patJetsAK4.userData.userInts.src += ['pileupJetIdEvaluator:cutbasedId','pileupJetIdEvaluator:fullId']
process.out.outputCommands += ['keep *_pileupJetIdEvaluator_*_*']

#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#QGTagger

process.load('RecoJets.JetProducers.QGTagger_cfi')
patJetsAK4.userData.userFloats.src += ['QGTagger:qgLikelihood']
process.out.outputCommands += ['keep *_QGTagger_*_*']

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
#ECF

process.load('RecoJets.JetProducers.ECF_cfi')

process.ECFCA8 = process.ECF.clone()
process.ECFCA8.src = cms.InputTag("ca8PFJetsCHS")
process.ECFCA8.cone = cms.double(0.8)

patJetsCA8.userData.userFloats.src += ['ECFCA8:ecf1','ECFCA8:ecf2','ECFCA8:ecf3']
process.out.outputCommands += ['keep *_ECFCA8_*_*']

process.ECFAK8 = process.ECFCA8.clone()
process.ECFAK8.src = cms.InputTag("ak8PFJetsCHS")

patJetsAK8.userData.userFloats.src += ['ECFAK8:ecf1','ECFAK8:ecf2','ECFAK8:ecf3']
process.out.outputCommands += ['keep *_ECFAK8_*_*']

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

patJetsCA8.userData.userFloats.src += ['ca8PFJetsCHSPrunedMass','ca8PFJetsCHSSoftDropMass','ca8PFJetsCHSTrimmedMass','ca8PFJetsCHSFilteredMass']
process.out.outputCommands += ['keep *_ca8PFJetsCHSPrunedMass_*_*',
                               'keep *_ca8PFJetsCHSSoftDropMass_*_*',
                               'keep *_ca8PFJetsCHSTrimmedMass_*_*',
                               'keep *_ca8PFJetsCHSFilteredMass_*_*']

patJetsAK8.userData.userFloats.src += ['ak8PFJetsCHSPrunedMass','ak8PFJetsCHSSoftDropMass','ak8PFJetsCHSTrimmedMass','ak8PFJetsCHSFilteredMass']
process.out.outputCommands += ['keep *_ak8PFJetsCHSPrunedMass_*_*',
                               'keep *_ak8PFJetsCHSSoftDropMass_*_*',
                               'keep *_ak8PFJetsCHSTrimmedMass_*_*',
                               'keep *_ak8PFJetsCHSFilteredMass_*_*']

process.cmsTopTagPFJetsCHSMassCA8 = process.ca8PFJetsCHSPrunedMass.clone()
process.cmsTopTagPFJetsCHSMassCA8.src = cms.InputTag("ca8PFJetsCHS")
process.cmsTopTagPFJetsCHSMassCA8.matched = cms.InputTag("cmsTopTagPFJetsCHS")

patJetsCA8.userData.userFloats.src += ['cmsTopTagPFJetsCHSMassCA8']
process.out.outputCommands += ['keep *_cmsTopTagPFJetsCHSMassCA8_*_*']

process.cmsTopTagPFJetsCHSMassAK8 = process.cmsTopTagPFJetsCHSMassCA8.clone()
process.cmsTopTagPFJetsCHSMassAK8.src = cms.InputTag("ak8PFJetsCHS")
process.cmsTopTagPFJetsCHSMassAK8.matched = cms.InputTag("cmsTopTagPFJetsCHS")

patJetsAK8.userData.userFloats.src += ['cmsTopTagPFJetsCHSMassAK8']
process.out.outputCommands += ['keep *_cmsTopTagPFJetsCHSMassAK8_*_*']

####################################################################################################

## ------------------------------------------------------
#  In addition you usually want to change the following
#  parameters:
## ------------------------------------------------------
#
#   process.GlobalTag.globaltag =  ...    ##  (according to https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideFrontierConditions)
#from Configuration.AlCa.GlobalTag import GlobalTag
#process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:run2_mc')
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
