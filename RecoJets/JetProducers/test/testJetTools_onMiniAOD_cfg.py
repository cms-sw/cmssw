## import skeleton process
from PhysicsTools.PatAlgos.patTemplate_cfg import *
#process.Tracer = cms.Service('Tracer')

from PhysicsTools.PatAlgos.tools.jetTools import updateJetCollection

updateJetCollection(
    process,
    labelName = 'AK8PFCHS',
    jetSource = cms.InputTag('slimmedJetsAK8'),
    algo = 'ak8',
    rParam = 0.8,
    jetCorrections = ('AK8PFchs', cms.vstring(['L1FastJet', 'L2Relative', 'L3Absolute']), 'None')
    )

updateJetCollection(
    process,
    labelName = 'AK4PFCHS',
    jetSource = cms.InputTag('slimmedJets'),
    algo = 'ak4',
    rParam = 0.4,
    jetCorrections = ('AK4PFchs', cms.vstring(['L1FastJet', 'L2Relative', 'L3Absolute']), 'None'),
    )

patJetsAK4 = process.updatedPatJetsAK4PFCHS
patJetsAK8 = process.updatedPatJetsAK8PFCHS

process.out.outputCommands += ['keep *_updatedPatJetsAK4PFCHS_*_*',
                               'keep *_updatedPatJetsAK8PFCHS_*_*']

####################################################################################################
#THE JET TOOLBOX

#load the various tools

#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#pileupJetID

process.load('RecoJets.JetProducers.PileupJetID_cfi')
patAlgosToolsTask.add(process.pileUpJetIDTask)
process.pileupJetIdCalculator.jets=cms.InputTag("slimmedJets")
process.pileupJetIdCalculator.inputIsCorrected=True
process.pileupJetIdCalculator.applyJec=True
process.pileupJetIdCalculator.vertexes=cms.InputTag("offlineSlimmedPrimaryVertices")
process.pileupJetIdEvaluator.jets=process.pileupJetIdCalculator.jets
process.pileupJetIdEvaluator.inputIsCorrected=process.pileupJetIdCalculator.inputIsCorrected
process.pileupJetIdEvaluator.applyJec=process.pileupJetIdCalculator.applyJec
process.pileupJetIdEvaluator.vertexes=process.pileupJetIdCalculator.vertexes
patJetsAK4.userData.userFloats.src += ['pileupJetIdEvaluator:fullDiscriminant']
patJetsAK4.userData.userInts.src += ['pileupJetIdEvaluator:cutbasedId','pileupJetIdEvaluator:fullId']
process.out.outputCommands += ['keep *_pileupJetIdEvaluator_*_*']

#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#QGTagger

process.load('RecoJets.JetProducers.QGTagger_cfi')
patAlgosToolsTask.add(process.QGTagger)
process.QGTagger.srcJets=cms.InputTag("slimmedJets")
process.QGTagger.srcVertexCollection=cms.InputTag("offlineSlimmedPrimaryVertices")
patJetsAK4.userData.userFloats.src += ['QGTagger:qgLikelihood']
process.out.outputCommands += ['keep *_QGTagger_*_*']

#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#Njettiness

process.load('RecoJets.JetProducers.nJettinessAdder_cfi')
patAlgosToolsTask.add(process.Njettiness)
process.NjettinessAK8 = process.Njettiness.clone()
patAlgosToolsTask.add(process.NjettinessAK8)
process.NjettinessAK8.src = cms.InputTag("slimmedJetsAK8")

patJetsAK8.userData.userFloats.src += ['NjettinessAK8:tau1','NjettinessAK8:tau2','NjettinessAK8:tau3']
process.out.outputCommands += ['keep *_NjettinessAK8_*_*']

#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#ECF

process.load('RecoJets.JetProducers.ECF_cff')
patAlgosToolsTask.add(process.ecf)
process.ecfAK8 = process.ecf.clone()
patAlgosToolsTask.add(process.ecfAK8)
#process.ecfAK8.cone = cms.double(0.8)
process.ecfAK8.src = cms.InputTag("slimmedJetsAK8")

patJetsAK8.userData.userFloats.src += ['ecfAK8:ecf1','ecfAK8:ecf2','ecfAK8:ecf3']
process.out.outputCommands += ['keep *_ecfAK8_*_*']

#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#QJetsAdder

process.RandomNumberGeneratorService = cms.Service("RandomNumberGeneratorService",
                                                   QJetsAdderAK8 = cms.PSet(initialSeed = cms.untracked.uint32(31)))

process.load('RecoJets.JetProducers.qjetsadder_cfi')
patAlgosToolsTask.add(process.QJetsAdder)
process.QJetsAdderAK8 = process.QJetsAdder.clone()
patAlgosToolsTask.add(process.QJetsAdderAK8)
process.QJetsAdderAK8.src = cms.InputTag("slimmedJetsAK8")
process.QJetsAdderAK8.jetRad = cms.double(0.8)
process.QJetsAdderAK8.jetAlgo = cms.string('AK')

patJetsAK8.userData.userFloats.src += ['QJetsAdderAK8:QjetsVolatility']
process.out.outputCommands += ['keep *_QJetsAdderAK8_*_*']
                                   
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#Grooming valueMaps

from RecoJets.Configuration.RecoPFJets_cff import ak8PFJetsCHSPruned, ak8PFJetsCHSSoftDrop, ak8PFJetsCHSTrimmed, ak8PFJetsCHSFiltered
process.ak8PFJetsCHSPruned = ak8PFJetsCHSPruned.clone()
patAlgosToolsTask.add(process.ak8PFJetsCHSPruned)
process.ak8PFJetsCHSSoftDrop = ak8PFJetsCHSSoftDrop.clone()
patAlgosToolsTask.add(process.ak8PFJetsCHSSoftDrop)
process.ak8PFJetsCHSTrimmed = ak8PFJetsCHSTrimmed.clone()
patAlgosToolsTask.add(process.ak8PFJetsCHSTrimmed)
process.ak8PFJetsCHSFiltered = ak8PFJetsCHSFiltered.clone()
patAlgosToolsTask.add(process.ak8PFJetsCHSFiltered)
process.ak8PFJetsCHSPruned.src = cms.InputTag("packedPFCandidates")
process.ak8PFJetsCHSSoftDrop.src = cms.InputTag("packedPFCandidates")
process.ak8PFJetsCHSTrimmed.src = cms.InputTag("packedPFCandidates")
process.ak8PFJetsCHSFiltered.src = cms.InputTag("packedPFCandidates")
from RecoJets.Configuration.RecoPFJets_cff import ak8PFJetsCHSPrunedMass, ak8PFJetsCHSSoftDropMass, ak8PFJetsCHSTrimmedMass, ak8PFJetsCHSFilteredMass
process.ak8PFJetsCHSPrunedMass = ak8PFJetsCHSPrunedMass.clone()
patAlgosToolsTask.add(process.ak8PFJetsCHSPrunedMass)
process.ak8PFJetsCHSSoftDropMass = ak8PFJetsCHSSoftDropMass.clone()
patAlgosToolsTask.add(process.ak8PFJetsCHSSoftDropMass)
process.ak8PFJetsCHSTrimmedMass = ak8PFJetsCHSTrimmedMass.clone()
patAlgosToolsTask.add(process.ak8PFJetsCHSTrimmedMass)
process.ak8PFJetsCHSFilteredMass = ak8PFJetsCHSFilteredMass.clone()
patAlgosToolsTask.add(process.ak8PFJetsCHSFilteredMass)
process.ak8PFJetsCHSPrunedMass.src = cms.InputTag("slimmedJetsAK8")
process.ak8PFJetsCHSSoftDropMass.src = cms.InputTag("slimmedJetsAK8")
process.ak8PFJetsCHSTrimmedMass.src = cms.InputTag("slimmedJetsAK8")
process.ak8PFJetsCHSFilteredMass.src = cms.InputTag("slimmedJetsAK8")

patJetsAK8.userData.userFloats.src += ['ak8PFJetsCHSPrunedMass','ak8PFJetsCHSSoftDropMass','ak8PFJetsCHSTrimmedMass','ak8PFJetsCHSFilteredMass']
process.out.outputCommands += ['keep *_ak8PFJetsCHSPrunedMass_*_*',
                               'keep *_ak8PFJetsCHSSoftDropMass_*_*',
                               'keep *_ak8PFJetsCHSTrimmedMass_*_*',
                               'keep *_ak8PFJetsCHSFilteredMass_*_*']

from RecoJets.JetProducers.caTopTaggers_cff import cmsTopTagPFJetsCHS
process.cmsTopTagPFJetsCHS = cmsTopTagPFJetsCHS.clone()
patAlgosToolsTask.add(process.cmsTopTagPFJetsCHS)
process.cmsTopTagPFJetsCHS.src = cms.InputTag("packedPFCandidates")
process.cmsTopTagPFJetsCHSMassAK8 = process.ak8PFJetsCHSPrunedMass.clone()
patAlgosToolsTask.add(process.cmsTopTagPFJetsCHSMassAK8)
process.cmsTopTagPFJetsCHSMassAK8.src = cms.InputTag("slimmedJetsAK8")
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
import PhysicsTools.PatAlgos.patInputFiles_cff
from PhysicsTools.PatAlgos.patInputFiles_cff import filesRelValTTbarPileUpMINIAODSIM
process.source.fileNames = filesRelValTTbarPileUpMINIAODSIM
#                                         ##
process.maxEvents.input = 5
#                                         ##
#   process.out.outputCommands = [ ... ]  ##  (e.g. taken from PhysicsTools/PatAlgos/python/patEventContent_cff.py)
#                                         ##
process.out.fileName = 'testJetTools.root'
#                                         ##
#   process.options.wantSummary = False   ##  (to suppress the long output at the end of the job)

