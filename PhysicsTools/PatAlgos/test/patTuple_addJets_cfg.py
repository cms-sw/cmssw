## import skeleton process
from PhysicsTools.PatAlgos.patTemplate_cfg import *
## switch to uncheduled mode
process.options.allowUnscheduled = cms.untracked.bool(True)
#process.Tracer = cms.Service("Tracer")

process.load("PhysicsTools.PatAlgos.producersLayer1.patCandidates_cff")
process.load("PhysicsTools.PatAlgos.selectionLayer1.selectedPatCandidates_cff")

from PhysicsTools.PatAlgos.tools.metTools import addMETCollection
#addMETCollection(process, labelName='patMETCalo', metSource='met')
addMETCollection(process, labelName='patMETPF', metSource='pfMetT1')
#addMETCollection(process, labelName='patMETTC', metSource='tcMet') # FIXME: removed from RECO/AOD; needs functionality to add to processing

## uncomment the following line to add different jet collections
## to the event content
from PhysicsTools.PatAlgos.tools.jetTools import addJetCollection
from PhysicsTools.PatAlgos.tools.jetTools import switchJetCollection

## uncomment the following lines to add ak4PFJetsCHS to your PAT output
labelAK4PFCHS = 'AK4PFCHS'
postfixAK4PFCHS = 'Copy'
addJetCollection(
   process,
   postfix   = postfixAK4PFCHS,
   labelName = labelAK4PFCHS,
   jetSource = cms.InputTag('ak4PFJetsCHS'),
   jetCorrections = ('AK4PFchs', cms.vstring(['L1FastJet', 'L2Relative', 'L3Absolute']), 'Type-2')
   )
process.out.outputCommands.append( 'drop *_selectedPatJets%s%s_caloTowers_*'%( labelAK4PFCHS, postfixAK4PFCHS ) )

# uncomment the following lines to add ak4PFJets to your PAT output
switchJetCollection(
   process,
   jetSource = cms.InputTag('ak4PFJets'),
   jetCorrections = ('AK4PF', cms.vstring(['L1FastJet', 'L2Relative', 'L3Absolute']), 'Type-1'),
   btagDiscriminators = [
       'pfJetBProbabilityBJetTags'
     , 'pfJetProbabilityBJetTags'
     , 'pfTrackCountingHighPurBJetTags'
     , 'pfTrackCountingHighEffBJetTags'
     , 'pfSimpleSecondaryVertexHighEffBJetTags'
     , 'pfSimpleSecondaryVertexHighPurBJetTags'
     , 'pfCombinedInclusiveSecondaryVertexV2BJetTags'
     ]
   )
process.out.outputCommands.append( 'drop *_selectedPatJets_caloTowers_*' )

# uncomment the following lines to add ak8PFJetsCHSSoftDrop to your PAT output
labelAK8PFCHSSoftDrop = 'AK8PFCHSSoftDrop'
addJetCollection(
   process,
   labelName = labelAK8PFCHSSoftDrop,
   jetSource = cms.InputTag('ak8PFJetsCHSSoftDrop',''),
   algo = 'AK',
   rParam = 0.8,
   genJetCollection = cms.InputTag('ak8GenJets'),
   jetCorrections = ('AK8PFchs', cms.vstring(['L1FastJet', 'L2Relative', 'L3Absolute']), 'None'),
   btagDiscriminators = ['None'], # turn-off b tagging
   getJetMCFlavour = False # jet flavor needs to be disabled for groomed fat jets
   )
process.out.outputCommands.append( 'keep *_selectedPatJets%s_pfCandidates_*'%( labelAK8PFCHSSoftDrop ) )
process.out.outputCommands.append( 'drop *_selectedPatJets%s_caloTowers_*'%( labelAK8PFCHSSoftDrop ) )

# uncomment the following lines to switch to ak4CaloJets in your PAT output
labelAK4Calo = 'AK4Calo'
addJetCollection(
   process,
   labelName = labelAK4Calo,
   jetSource = cms.InputTag('ak4CaloJets'),
   jetCorrections = ('AK7Calo', cms.vstring(['L1FastJet', 'L2Relative', 'L3Absolute']), 'Type-1'), # FIXME: Use proper JECs, as soon as available
   btagDiscriminators = [
       'pfJetBProbabilityBJetTags'
     , 'pfJetProbabilityBJetTags'
     , 'pfTrackCountingHighPurBJetTags'
     , 'pfTrackCountingHighEffBJetTags'
     , 'pfSimpleSecondaryVertexHighEffBJetTags'
     , 'pfSimpleSecondaryVertexHighPurBJetTags'
     , 'pfCombinedInclusiveSecondaryVertexV2BJetTags'
     ]
   )
process.out.outputCommands.append( 'drop *_selectedPatJets%s_pfCandidates_*'%( labelAK4Calo ) )
## JetID works only with RECO input for the CaloTowers (s. below for 'process.source.fileNames')
#process.patJets.addJetID=True
#process.load("RecoJets.JetProducers.ak4JetID_cfi")
#process.patJets.jetIDMap="ak4JetID"
process.patJetsAK4Calo.useLegacyJetMCFlavour=True # Need to use legacy flavour since the new flavour requires jet constituents which are dropped for CaloJets from AOD

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
#from PhysicsTools.PatAlgos.patInputFiles_cff import filesRelValProdTTbarGENSIMRECO
#process.source.fileNames = filesRelValProdTTbarGENSIMRECO
#                                         ##
process.maxEvents.input = 10
#                                         ##
#   process.out.outputCommands = [ ... ]  ##  (e.g. taken from PhysicsTools/PatAlgos/python/patEventContent_cff.py)
#                                         ##
process.out.fileName = 'patTuple_addJets.root'
#                                         ##
#   process.options.wantSummary = False   ##  (to suppress the long output at the end of the job)
