## import skeleton process
from PhysicsTools.PatAlgos.patTemplate_cfg import *

## to run in un-scheduled mode uncomment the following lines
process.load("PhysicsTools.PatAlgos.producersLayer1.patCandidates_cff")
patAlgosToolsTask.add(process.patCandidatesTask)

process.load("PhysicsTools.PatAlgos.selectionLayer1.selectedPatCandidates_cff")
patAlgosToolsTask.add(process.selectedPatCandidatesTask)

## uncomment the following line to add different jet collections
## to the event content
from PhysicsTools.PatAlgos.tools.jetTools import addJetCollection

# b-tag discriminators
btagDiscriminators = [
     # legacy framework (no longer supported, work with RECO/AOD but not MiniAOD)
     'jetBProbabilityBJetTags'
    ,'jetProbabilityBJetTags'
    ,'positiveOnlyJetBProbabilityBJetTags'
    ,'positiveOnlyJetProbabilityBJetTags'
    ,'negativeOnlyJetBProbabilityBJetTags'
    ,'negativeOnlyJetProbabilityBJetTags'
    ,'trackCountingHighPurBJetTags'
    ,'trackCountingHighEffBJetTags'
    ,'negativeTrackCountingHighEffBJetTags'
    ,'negativeTrackCountingHighPurBJetTags'
    ,'simpleSecondaryVertexHighEffBJetTags'
    ,'simpleSecondaryVertexHighPurBJetTags'
    ,'negativeSimpleSecondaryVertexHighEffBJetTags'
    ,'negativeSimpleSecondaryVertexHighPurBJetTags'
    ,'combinedSecondaryVertexV2BJetTags'
    ,'positiveCombinedSecondaryVertexV2BJetTags'
    ,'negativeCombinedSecondaryVertexV2BJetTags'
    ,'simpleInclusiveSecondaryVertexHighEffBJetTags'
    ,'simpleInclusiveSecondaryVertexHighPurBJetTags'
    ,'negativeSimpleInclusiveSecondaryVertexHighEffBJetTags'
    ,'negativeSimpleInclusiveSecondaryVertexHighPurBJetTags'
    ,'doubleSecondaryVertexHighEffBJetTags'
    ,'combinedInclusiveSecondaryVertexV2BJetTags'
    ,'positiveCombinedInclusiveSecondaryVertexV2BJetTags'
    ,'negativeCombinedInclusiveSecondaryVertexV2BJetTags'
    ,'combinedMVAV2BJetTags'
    ,'negativeCombinedMVAV2BJetTags'
    ,'positiveCombinedMVAV2BJetTags'
     # new candidate-based framework (supported with RECO/AOD/MiniAOD)
    ,'pfJetBProbabilityBJetTags'
    ,'pfJetProbabilityBJetTags'
    ,'pfPositiveOnlyJetBProbabilityBJetTags'
    ,'pfPositiveOnlyJetProbabilityBJetTags'
    ,'pfNegativeOnlyJetBProbabilityBJetTags'
    ,'pfNegativeOnlyJetProbabilityBJetTags'
    ,'pfTrackCountingHighPurBJetTags'
    ,'pfTrackCountingHighEffBJetTags'
    ,'pfNegativeTrackCountingHighPurBJetTags'
    ,'pfNegativeTrackCountingHighEffBJetTags'
    ,'pfSimpleSecondaryVertexHighEffBJetTags'
    ,'pfSimpleSecondaryVertexHighPurBJetTags'
    ,'pfNegativeSimpleSecondaryVertexHighEffBJetTags'
    ,'pfNegativeSimpleSecondaryVertexHighPurBJetTags'
    ,'pfSimpleInclusiveSecondaryVertexHighEffBJetTags'
    ,'pfSimpleInclusiveSecondaryVertexHighPurBJetTags'
    ,'pfNegativeSimpleInclusiveSecondaryVertexHighEffBJetTags'
    ,'pfNegativeSimpleInclusiveSecondaryVertexHighPurBJetTags'
    ,'pfCombinedSecondaryVertexV2BJetTags'
    ,'pfPositiveCombinedSecondaryVertexV2BJetTags'
    ,'pfNegativeCombinedSecondaryVertexV2BJetTags'
    ,'pfCombinedInclusiveSecondaryVertexV2BJetTags'
    ,'pfPositiveCombinedInclusiveSecondaryVertexV2BJetTags'
    ,'pfNegativeCombinedInclusiveSecondaryVertexV2BJetTags'
    ,'pfGhostTrackBJetTags'
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
    ,'pfCombinedMVAV2BJetTags'
    ,'pfNegativeCombinedMVAV2BJetTags'
    ,'pfPositiveCombinedMVAV2BJetTags'
     # CTagging
    ,'pfCombinedCvsLJetTags'
    ,'pfCombinedCvsBJetTags'
     # ChargeTagging
    ,'pfChargeBJetTags'
     #Deep Flavour
    ,'pfDeepCSVJetTags:probb'
    ,'pfDeepCSVJetTags:probc'
    ,'pfDeepCSVJetTags:probudsg'
    ,'pfDeepCSVJetTags:probbb'
     # DeepCMVA
    ,'pfDeepCMVAJetTags:probb'
    ,'pfDeepCMVAJetTags:probc'
    ,'pfDeepCMVAJetTags:probudsg'
    ,'pfDeepCMVAJetTags:probbb'
    ,'pfDeepCMVAJetTags:probcc'
]

# uncomment the following lines to add ak4PFJets with new b-tags to your PAT output
addJetCollection(
   process,
   labelName = 'AK4PF',
   jetSource = cms.InputTag('ak4PFJets'),
   jetCorrections = ('AK4PF', cms.vstring(['L1FastJet', 'L2Relative', 'L3Absolute']), 'Type-2'),
   btagDiscriminators = btagDiscriminators
)
process.patJetsAK4PF.addTagInfos = True

# uncomment the following lines to add ak8PFJetsCHS with new b-tags to your PAT output
addJetCollection(
   process,
   labelName = 'AK8PFCHS',
   jetSource = cms.InputTag('ak8PFJetsCHS'),
   jetCorrections = ('AK8PFchs', cms.vstring(['L1FastJet', 'L2Relative', 'L3Absolute']), 'Type-2'),
   algo = 'AK',
   rParam = 0.8,
   btagDiscriminators = ['pfBoostedDoubleSecondaryVertexAK8BJetTags']
)
process.patJetsAK8PFCHS.addTagInfos = True

# uncomment the following lines to add subjets of ak8PFJetsCHSSoftDrop with new b-tags to your PAT output
from pdb import set_trace
addJetCollection(
   process,
   labelName = 'AK8PFCHSSoftDropSubjets',
   jetSource = cms.InputTag('ak8PFJetsCHSSoftDrop','SubJets'),
   jetCorrections = ('AK4PFchs', cms.vstring(['L1FastJet', 'L2Relative', 'L3Absolute']), 'Type-2'), # Using AK4 JECs for subjets which might not be completely appropriate
   algo = 'AK',  # needed for subjet flavor clustering
   btagDiscriminators = btagDiscriminators,
   explicitJTA = True,  # needed for subjet b tagging
   svClustering = True, # needed for subjet b tagging
   fatJets = cms.InputTag("ak8PFJetsCHS"),               # needed for subjet flavor clustering
   groomedFatJets = cms.InputTag("ak8PFJetsCHSSoftDrop"), # needed for subjet flavor clustering
   rParam = 0.8, # needed for subjet flavor clustering
)
process.patJetsAK8PFCHSSoftDropSubjets.addTagInfos = True

## JetID works only with RECO input for the CaloTowers (s. below for 'process.source.fileNames')
#process.patJets.addJetID=True
#process.load("RecoJets.JetProducers.ak4JetID_cfi")
#process.patJets.jetIDMap="ak4JetID"
process.out.outputCommands.append( 'drop *_selectedPatJetsAK4PF_caloTowers_*' )

## ------------------------------------------------------
#  In addition you usually want to change the following
#  parameters:
## ------------------------------------------------------
#
#process.GlobalTag.globaltag =  'MCRUN1_74_V2::All'     ##  (according to https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideFrontierConditions)
#                                         ##
## switch to RECO input
from PhysicsTools.PatAlgos.patInputFiles_cff import filesRelValProdTTbarAODSIM
process.source.fileNames = filesRelValProdTTbarAODSIM
#from PhysicsTools.PatAlgos.patInputFiles_cff import filesRelValTTbarGENSIMRECO
#process.source.fileNames = filesRelValTTbarGENSIMRECO
#                                         ##
process.maxEvents.input = 10
#                                         ##
#   process.out.outputCommands = [ ... ]  ##  (e.g. taken from PhysicsTools/PatAlgos/python/patEventContent_cff.py)
#                                         ##
process.out.fileName = 'patTuple_addBTagging.root'
#                                         ##
process.options.wantSummary = False   ##  (to suppress the long output at the end of the job)
