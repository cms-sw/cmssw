## import skeleton process
from PhysicsTools.PatAlgos.patTemplate_cfg import *
## switch to uncheduled mode
process.options.allowUnscheduled = cms.untracked.bool(True)
#process.Tracer = cms.Service("Tracer")

process.load("PhysicsTools.PatAlgos.producersLayer1.patCandidates_cff")
process.load("PhysicsTools.PatAlgos.selectionLayer1.selectedPatCandidates_cff")

from PhysicsTools.PatAlgos.tools.metTools import addMETCollection
addMETCollection(process, labelName='patMETCalo', metSource='met')
addMETCollection(process, labelName='patMETPF', metSource='pfType1CorrectedMet')
addMETCollection(process, labelName='patMETTC', metSource='tcMet')

## uncomment the following line to add different jet collections
## to the event content
from PhysicsTools.PatAlgos.tools.jetTools import addJetCollection
from PhysicsTools.PatAlgos.tools.jetTools import switchJetCollection

## uncomment the following lines to add ak4PFJetsCHS to your PAT output
postfixAK5PFCHS = 'Copy'
addJetCollection(
   process,
   postfix   = postfixAK5PFCHS,
   labelName = 'AK5PFCHS',
   jetSource = cms.InputTag('ak4PFJetsCHS'),
   jetCorrections = ('AK5PFchs', cms.vstring(['L1FastJet', 'L2Relative', 'L3Absolute']), 'Type-2')
   )
process.out.outputCommands.append( 'drop *_selectedPatJetsAK5PFCHS%s_caloTowers_*'%( postfixAK5PFCHS ) )

# uncomment the following lines to add ak4PFJets to your PAT output
addJetCollection(
   process,
   labelName = 'AK5PF',
   jetSource = cms.InputTag('ak4PFJets'),
   jetCorrections = ('AK5PF', cms.vstring(['L1FastJet', 'L2Relative', 'L3Absolute']), 'Type-1'),
   btagDiscriminators = [
       'jetBProbabilityBJetTags'
     , 'jetProbabilityBJetTags'
     , 'trackCountingHighPurBJetTags'
     , 'trackCountingHighEffBJetTags'
     , 'simpleSecondaryVertexHighEffBJetTags'
     , 'simpleSecondaryVertexHighPurBJetTags'
     , 'combinedSecondaryVertexBJetTags'
     ],
   )
process.out.outputCommands.append( 'drop *_selectedPatJetsAK5PF_caloTowers_*' )

# uncomment the following lines to switch to ak4CaloJets in your PAT output
switchJetCollection(
   process,
   jetSource = cms.InputTag('ak4CaloJets'),
   jetCorrections = ('AK5Calo', cms.vstring(['L1Offset', 'L2Relative', 'L3Absolute']), 'Type-1'),
   btagDiscriminators = [
       'jetBProbabilityBJetTags'
     , 'jetProbabilityBJetTags'
     , 'trackCountingHighPurBJetTags'
     , 'trackCountingHighEffBJetTags'
     , 'simpleSecondaryVertexHighEffBJetTags'
     , 'simpleSecondaryVertexHighPurBJetTags'
     , 'combinedSecondaryVertexBJetTags'
     ],
   )
process.patJets.addJetID=True
process.patJets.jetIDMap="ak4JetID"
process.out.outputCommands.append( 'keep *_selectedPatJets_caloTowers_*' )
process.out.outputCommands.append( 'drop *_selectedPatJets_pfCandidates_*' )

#print process.out.outputCommands

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
process.maxEvents.input = 10
#                                         ##
#   process.out.outputCommands = [ ... ]  ##  (e.g. taken from PhysicsTools/PatAlgos/python/patEventContent_cff.py)
#                                         ##
process.out.fileName = 'patTuple_addJets.root'
#                                         ##
#   process.options.wantSummary = False   ##  (to suppress the long output at the end of the job)
