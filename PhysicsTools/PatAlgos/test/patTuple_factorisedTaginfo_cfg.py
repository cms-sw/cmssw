## import skeleton process
from PhysicsTools.PatAlgos.patTemplate_cfg import *

from PhysicsTools.PatAlgos.tools.jetTools import *

addJetCollection(process,cms.InputTag('ak7CaloJets'),
                 'AK7', 'Calo',
                 doJTA        = True,
#                 doBTagging   = False,
                 doBTagging   = True,
#                 jetCorrLabel = ('AK7CaloJets', ['L2Relative', 'L3Absolute']),
                 doType1MET   = True,
                 doL1Cleaning = True,
                 doL1Counters = False,
                 genJetCollection=cms.InputTag("ak7GenJets"),

                 doJetID      = True,
                 jetIdLabel   = "ak7"


#		 ,btagInfo    = ['impactParameterTagInfos','secondaryVertexTagInfos','softMuonTagInfos']



#######		  ,btagInfo = ['impactParameterTagInfos']
#                 ,btagInfo = ['softMuonTagInfos']
#                 ,btagInfo = ['impactParameterTagInfos','secondaryVertexTagInfos']
#   		 ,btagInfo = ['impactParameterTagInfos','secondaryVertexTagInfos','secondaryVertexNegativeTagInfos']


#		 ,btagdiscriminators=['jetBProbabilityBJetTags', 'jetProbabilityBJetTags', 'trackCountingHighPurBJetTags','trackCountingHighEffBJetTags',
#	'simpleSecondaryVertexHighEffBJetTags','simpleSecondaryVertexHighPurBJetTags','combinedSecondaryVertexBJetTags','combinedSecondaryVertexMVABJetTags','softMuonBJetTags','softMuonByPtBJetTags','softMuonByIP3dBJetTags']

#		,btagdiscriminators=['softMuonBJetTags','softMuonByPtBJetTags','jetBProbabilityBJetTags', 'jetProbabilityBJetTags', 'trackCountingHighPurBJetTags','trackCountingHighEffBJetTags']


#                ,btagdiscriminators=['jetBProbabilityBJetTags','jetProbabilityBJetTags','trackCountingHighPurBJetTags','trackCountingHighEffBJetTags']
#                ,btagdiscriminators=['softMuonBJetTags','softMuonByPtBJetTags','softMuonByIP3dBJetTags']
#                ,btagdiscriminators=['combinedSecondaryVertexBJetTags','combinedSecondaryVertexMVABJetTags']
###                ,btagdiscriminators=['simpleSecondaryVertexHighEffBJetTags','simpleSecondaryVertexHighPurBJetTags']
##                ,btagdiscriminators=['simpleSecondaryVertexNegativeHighEffBJetTags','simpleSecondaryVertexNegativeHighPurBJetTags','negativeTrackCountingHighEffJetTags','negativeTrackCountingHighPurJetTags']
#		 ,btagdiscriminators=['negativeTrackCountingHighEffJetTags','negativeTrackCountingHighPurJetTags']


	)

switchJetCollection(process,cms.InputTag('ak5PFJets'),
                 doJTA        = True,
#                 doBTagging   = False,
                 doBTagging   = True,
#                 jetCorrLabel = None,
                 doType1MET   = True,
                 genJetCollection=cms.InputTag("ak5GenJets"),
                 doJetID      = True

#                ,btagInfo    = ['impactParameterTagInfos','secondaryVertexTagInfos','softMuonTagInfos']

#######                 ,btagInfo = ['impactParameterTagInfos']
###                 ,btagInfo = ['softMuonTagInfos']
#                 ,btagInfo = ['impactParameterTagInfos','secondaryVertexTagInfos']
#                 ,btagInfo = ['impactParameterTagInfos','secondaryVertexTagInfos','secondaryVertexNegativeTagInfos']



#                ,btagdiscriminators=['jetBProbabilityBJetTags', 'jetProbabilityBJetTags', 'trackCountingHighPurBJetTags','trackCountingHighEffBJetTags',
#       'simpleSecondaryVertexHighEffBJetTags','simpleSecondaryVertexHighPurBJetTags','combinedSecondaryVertexBJetTags','combinedSecondaryVertexMVABJetTags','softMuonBJetTags','softMuonByPtBJetTags','softMuonByIP3dBJetTags']

#               ,btagdiscriminators=['softMuonBJetTags','softMuonByPtBJetTags','jetBProbabilityBJetTags', 'jetProbabilityBJetTags', 'trackCountingHighPurBJetTags','trackCountingHighEffBJetTags']


#                ,btagdiscriminators=['jetBProbabilityBJetTags','jetProbabilityBJetTags','trackCountingHighPurBJetTags','trackCountingHighEffBJetTags']
###                ,btagdiscriminators=['softMuonBJetTags','softMuonByPtBJetTags','softMuonByIP3dBJetTags']
#                ,btagdiscriminators=['combinedSecondaryVertexBJetTags','combinedSecondaryVertexMVABJetTags']
#                ,btagdiscriminators=['simpleSecondaryVertexHighEffBJetTags','simpleSecondaryVertexHighPurBJetTags']
#                ,btagdiscriminators=['simpleSecondaryVertexNegativeHighEffBJetTags','simpleSecondaryVertexNegativeHighPurBJetTags','negativeTrackCountingHighEffJetTags','negativeTrackCountingHighPurJetTags']
#                ,btagdiscriminators=['negativeTrackCountingHighEffJetTags','negativeTrackCountingHighPurJetTags']
#

                 )

#process.patJetsAK7Calo.addTagInfos = True
#process.patJets.addTagInfos = False

process.p = cms.Path(
    process.patDefaultSequence
    )

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
process.maxEvents.input = 100
#                                         ##
#   process.out.outputCommands = [ ... ]  ##  (e.g. taken from PhysicsTools/PatAlgos/python/patEventContent_cff.py)
#                                         ##
process.out.fileName = 'patTuple_factorisedTagInfo.root'
#                                         ##
#   process.options.wantSummary = False   ##  (to suppress the long output at the end of the job)



