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

# uncomment the following lines to add ak5PFJets with new b-tags to your PAT output
addJetCollection(
   process,
   labelName = 'AK5PF',
   jetSource = cms.InputTag('ak5PFJets'),
   jetCorrections = ('AK5PF', cms.vstring(['L1FastJet', 'L2Relative', 'L3Absolute']), 'Type-2'),
   btagDiscriminators = [
     'jetBProbabilityBJetTags',
     'jetProbabilityBJetTags',
     'trackCountingHighPurBJetTags',
     'trackCountingHighEffBJetTags',
     'simpleSecondaryVertexHighEffBJetTags',
     'simpleSecondaryVertexHighPurBJetTags',
     'combinedSecondaryVertexBJetTags',
     'combinedSecondaryVertexMVABJetTags',
     'softMuonBJetTags',
     'softMuonByPtBJetTags',
     'softMuonByIP3dBJetTags',
     'simpleSecondaryVertexNegativeHighEffBJetTags',
     'simpleSecondaryVertexNegativeHighPurBJetTags',
     'negativeTrackCountingHighEffJetTags',
     'negativeTrackCountingHighPurJetTags'
   ],
  )
process.patJetsAK5PF.addTagInfos = True
process.patJetsAK5PF.addJetID    = True
process.patJetsAK5PF.jetIDMap    = "ak5JetID"
process.out.outputCommands.append( 'drop *_selectedPatJetsAK5PF_caloTowers_*' )

## let it run
process.p = cms.Path(
    process.selectedPatCandidates
)

## ------------------------------------------------------
#  In addition you usually want to change the following
#  parameters:
## ------------------------------------------------------
#
#   process.GlobalTag.globaltag =  ...    ##  (according to https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideFrontierConditions)
#                                         ##
## switch to RECO input
from PhysicsTools.PatAlgos.patInputFiles_cff import filesRelValProdTTbarAODSIM
process.source.fileNames = filesRelValProdTTbarAODSIM
#                                         ##
process.maxEvents.input = 10
#                                         ##
#   process.out.outputCommands = [ ... ]  ##  (e.g. taken from PhysicsTools/PatAlgos/python/patEventContent_cff.py)
#                                         ##
process.out.fileName = 'patTuple_addBTagging.root'
#                                         ##
#   process.options.wantSummary = False   ##  (to suppress the long output at the end of the job)
