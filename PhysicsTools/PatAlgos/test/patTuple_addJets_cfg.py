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
#addJetCollection(process,cms.InputTag('ak7CaloJets'),
#                 'AK7', 'Calo',
#                 doJTA        = True,
#                 doBTagging   = False,
#                 jetCorrLabel = ('AK7Calo', cms.vstring(['L1Offset', 'L2Relative', 'L3Absolute'])),
#                 doType1MET   = True,
#                 doL1Cleaning = True,
#                 doL1Counters = False,
#                 genJetCollection=cms.InputTag("ak7GenJets"),
#                 doJetID      = True,
#                 jetIdLabel   = "ak7"
#                 )

## uncomment the following lines to add kt4CaloJets to your PAT output
#addJetCollection(process,cms.InputTag('kt4CaloJets'),
#                 'KT4', 'Calo',
#                 doJTA        = True,
#                 doBTagging   = True,
#                 jetCorrLabel = ('KT4Calo', cms.vstring(['L2Relative', 'L3Absolute'])),
#                 doType1MET   = True,
#                 doL1Cleaning = True,
#                 doL1Counters = False,
#                 genJetCollection=cms.InputTag("kt4GenJets"),
#                 doJetID      = True,
#                 jetIdLabel   = "kt4"
#                 )

## uncomment the following lines to add kt6CaloJets to your PAT output
addJetCollection(process,
                 labelName = 'AK5PF',
                 jetSource = cms.InputTag('ak5PFJets'),
                 jetCorrections = ('AK5PF', cms.vstring(['L1FastJet', 'L2Relative', 'L3Absolute']), 'Type-2'),
                 #btagDiscriminators = ['None'],
                 #btagInfos= ['None'],
                 jetTrackAssociation = True,
                 #outputModules   = ['out'],
                 )

## uncomment the following lines to add ak5PFJets to your PAT output
#switchJetCollection(process,cms.InputTag('ak5PFJets'),
#                 doJTA        = True,
#                 doBTagging   = True,
#                 jetCorrLabel = ('AK5PF', cms.vstring(['L1FastJet', 'L2Relative', 'L3Absolute'])),
#                 doType1MET   = True,
#                 genJetCollection=cms.InputTag("ak5GenJets"),
#                 doJetID      = True
#                 )

## let it run
#process.p = cms.Path(
#    process.patDefaultSequence
#)


#process.Tracer = cms.Service("Tracer")
process.p = cms.Path(
    process.selectedPatCandidates
    *process.selectedPatJetsAK5PF
    )



print process.out.outputCommands

## ------------------------------------------------------
#  In addition you usually want to change the following
#  parameters:
## ------------------------------------------------------
#
#   process.GlobalTag.globaltag =  ...    ##  (according to https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideFrontierConditions)
#                                         ##
#   process.source.fileNames =  ...       ##  (e.g. 'file:AOD.root')
#                                         ##
process.maxEvents.input = 100
#                                         ##
#   process.out.outputCommands = [ ... ]  ##  (e.g. taken from PhysicsTools/PatAlgos/python/patEventContent_cff.py)
#                                         ##
process.out.fileName = 'patTuple_addJets.root'
#                                         ##
#   process.options.wantSummary = False   ##  (to suppress the long output at the end of the job)
