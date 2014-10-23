## import skeleton process
from PhysicsTools.PatAlgos.patTemplate_cfg import *
## switch to uncheduled mode
process.options.allowUnscheduled = cms.untracked.bool(True)

## to run in un-scheduled mode uncomment the following lines
process.load("PhysicsTools.PatAlgos.producersLayer1.patCandidates_cff")
process.load("PhysicsTools.PatAlgos.selectionLayer1.selectedPatCandidates_cff")

## uncomment the following line to add different jet collections
## to the event content
from PhysicsTools.PatAlgos.tools.jetTools import addJetCollection

# uncomment the following lines to add ak4PFJets with new b-tags to your PAT output
addJetCollection(
   process,
   labelName = 'AK4PF',
   jetSource = cms.InputTag('ak4PFJets'),
   jetCorrections = ('AK4PF', cms.vstring(['L1FastJet', 'L2Relative', 'L3Absolute']), 'Type-2'), # FIXME: Use proper JECs, as soon as available
   btagDiscriminators = [
       'jetBProbabilityBJetTags'
      ,'jetProbabilityBJetTags'
      ,'trackCountingHighPurBJetTags'
      ,'trackCountingHighEffBJetTags'
      ,'negativeOnlyJetBProbabilityJetTags'
      ,'negativeOnlyJetProbabilityJetTags'
      ,'negativeTrackCountingHighEffJetTags'
      ,'negativeTrackCountingHighPurJetTags'
      ,'positiveOnlyJetBProbabilityJetTags'
      ,'positiveOnlyJetProbabilityJetTags'
      ,'simpleSecondaryVertexHighEffBJetTags'
      ,'simpleSecondaryVertexHighPurBJetTags'
      ,'simpleSecondaryVertexNegativeHighEffBJetTags'
      ,'simpleSecondaryVertexNegativeHighPurBJetTags'
      ,'pfCombinedSecondaryVertexBJetTags'
      ,'combinedSecondaryVertexBJetTags'
      ,'combinedSecondaryVertexPositiveBJetTags'
      ,'combinedInclusiveSecondaryVertexV2BJetTags'
      ,'combinedInclusiveSecondaryVertexV2PositiveBJetTags'
      ,'combinedInclusiveSecondaryVertexV2NegativeBJetTags'
      ,'combinedSecondaryVertexMVABJetTags'
      ,'combinedSecondaryVertexNegativeBJetTags'
      ,'softPFMuonBJetTags'
      ,'softPFMuonByPtBJetTags'
      ,'softPFMuonByIP3dBJetTags'
      ,'softPFMuonByIP2dBJetTags'
      ,'positiveSoftPFMuonBJetTags'
      ,'positiveSoftPFMuonByPtBJetTags'
      ,'positiveSoftPFMuonByIP3dBJetTags'
      ,'positiveSoftPFMuonByIP2dBJetTags'
      ,'negativeSoftPFMuonBJetTags'
      ,'negativeSoftPFMuonByPtBJetTags'
      ,'negativeSoftPFMuonByIP3dBJetTags'
      ,'negativeSoftPFMuonByIP2dBJetTags'
      ,'softPFElectronBJetTags'
      ,'softPFElectronByPtBJetTags'
      ,'softPFElectronByIP3dBJetTags'
      ,'softPFElectronByIP2dBJetTags'
      ,'positiveSoftPFElectronBJetTags'
      ,'positiveSoftPFElectronByPtBJetTags'
      ,'positiveSoftPFElectronByIP3dBJetTags'
      ,'positiveSoftPFElectronByIP2dBJetTags'
      ,'negativeSoftPFElectronBJetTags'
      ,'negativeSoftPFElectronByPtBJetTags'
      ,'negativeSoftPFElectronByIP3dBJetTags'
      ,'negativeSoftPFElectronByIP2dBJetTags'
      ,'simpleInclusiveSecondaryVertexHighEffBJetTags'
      ,'simpleInclusiveSecondaryVertexHighPurBJetTags'
      ,'doubleSecondaryVertexHighEffBJetTags'
      ,'combinedInclusiveSecondaryVertexBJetTags'
      ,'combinedInclusiveSecondaryVertexPositiveBJetTags'
      ,'combinedMVABJetTags'
      ,'positiveCombinedMVABJetTags'
      ,'negativeCombinedMVABJetTags'
    ],
  )
process.patJetsAK4PF.addTagInfos = True
## JetID works only with RECO input for the CaloTowers (s. below for 'process.source.fileNames')
#process.patJets.addJetID=True
#process.load("RecoJets.JetProducers.ak4JetID_cfi")
#process.patJets.jetIDMap="ak4JetID"
process.out.outputCommands.append( 'drop *_selectedPatJetsAK4PF_caloTowers_*' )

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
#from PhysicsTools.PatAlgos.patInputFiles_cff import filesRelValProdTTbarGENSIMRECO
#process.source.fileNames = filesRelValProdTTbarGENSIMRECO
#                                         ##
process.maxEvents.input = 10
#                                         ##
#   process.out.outputCommands = [ ... ]  ##  (e.g. taken from PhysicsTools/PatAlgos/python/patEventContent_cff.py)
#                                         ##
process.out.fileName = 'patTuple_addBTagging.root'
#                                         ##
#   process.options.wantSummary = False   ##  (to suppress the long output at the end of the job)
