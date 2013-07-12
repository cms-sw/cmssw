## import skeleton process
from PhysicsTools.PatAlgos.patTemplate_cfg import *
## switch to uncheduled mode
process.options.allowUnscheduled = cms.untracked.bool(True)

## to run in un-scheduled mode uncomment the following lines
process.load("PhysicsTools.PatAlgos.producersLayer1.patCandidates_cff")
process.load("PhysicsTools.PatAlgos.selectionLayer1.selectedPatCandidates_cff")
from PhysicsTools.PatAlgos.tools.metTools import addMETCollection

addMETCollection(process, labelName='patMETTC', metSource='tcMet')
addMETCollection(process, labelName='patMETPF', metSource='pfType1CorrectedMet')

## uncomment the following line to add different jet collections
## to the event content
from PhysicsTools.PatAlgos.tools.jetTools import addJetCollection
from PhysicsTools.PatAlgos.tools.jetTools import switchJetCollection


## uncomment the following lines to add ak5JPTJets to your PAT output
#addJetCollection(process,cms.InputTag('JetPlusTrackZSPCorJetAntiKt5'),
#                 'AK5', 'JPT',
#                 doJTA        = True,
#                 doBTagging   = True,
#                 jetCorrLabel = ('AK5JPT', cms.vstring(['L1Offset', 'L1JPTOffset', 'L2Relative', 'L3Absolute'])),
#                 doType1MET   = False,
#                 doL1Cleaning = False,
#                 doL1Counters = True,
#                 genJetCollection = cms.InputTag("ak5GenJets"),
#                 doJetID      = True,
#                 jetIdLabel   = "ak5"
#                 )

## uncomment the following lines to add ak7CaloJets to your PAT output
addJetCollection(
   process,
   labelName = 'AK7Calo',
   jetSource = cms.InputTag('ak7CaloJets'),
   jetCorrections = ('AK7Calo', cms.vstring(['L1Offset', 'L2Relative', 'L3Absolute']), 'Type-2'),
   )
process.patJetsAK7Calo.addJetID=True
process.patJetsAK7Calo.jetIDMap="ak7JetID"

## uncomment the following lines to add kt6CaloJets to your PAT output
postfixAK5Calo = 'Copy'
addJetCollection(
   process,
   postfix   = postfixAK5Calo,
   labelName = 'AK5Calo',
   jetSource = cms.InputTag('ak5CaloJets'),
   jetCorrections = ('AK5Calo', cms.vstring(['L1Offset', 'L2Relative', 'L3Absolute']), 'Type-2'),
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
getattr(process, 'patJetsAK5Calo' + postfixAK5Calo).addJetID=True
getattr(process, 'patJetsAK5Calo' + postfixAK5Calo).jetIDMap="ak5JetID"
process.out.outputCommands.append( 'drop *_selectedPatJetsAK5Calo%s_pfCandidates_*'%(postfixAK5Calo) )
#process.patJetsAK5Calo.addJetID=True
#process.patJetsAK5Calo.jetIDMap="ak5JetID"

## uncomment the following lines to add ak5PFJets to your PAT output
switchJetCollection(
   process,
   jetSource = cms.InputTag('ak5PFJets'),
   jetCorrections = ('AK5PF', cms.vstring(['L1FastJet', 'L2Relative', 'L3Absolute']), 'Type-2'),
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

## let it run
#process.p = cms.Path(
#    process.patDefaultSequence
#)

##process.Tracer = cms.Service("Tracer")
#process.p = cms.Path(
    #process.selectedPatCandidates
    #*process.selectedPatJetsAK5Calo
    #*process.selectedPatJetsAK7Calo
    #)



#print process.out.outputCommands

## ------------------------------------------------------
#  In addition you usually want to change the following
#  parameters:
## ------------------------------------------------------
#
#   process.GlobalTag.globaltag =  ...    ##  (according to https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideFrontierConditions)
#                                         ##
from PhysicsTools.PatAlgos.patInputFiles_cff import filesRelValProdTTbarGENSIMRECO
process.source.fileNames = filesRelValProdTTbarGENSIMRECO
#                                         ##
process.maxEvents.input = 10
#                                         ##
#   process.out.outputCommands = [ ... ]  ##  (e.g. taken from PhysicsTools/PatAlgos/python/patEventContent_cff.py)
#                                         ##
process.out.fileName = 'patTuple_addJets.root'
#                                         ##
#   process.options.wantSummary = False   ##  (to suppress the long output at the end of the job)
