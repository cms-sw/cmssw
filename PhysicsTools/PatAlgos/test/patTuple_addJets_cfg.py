## import skeleton process
from PhysicsTools.PatAlgos.patTemplate_cfg import *

##from PhysicsTools.PatAlgos.tools.coreTools import *
##removeMCMatching(process, ['All'])

## uncomment the following line to add tcMET to the event content
from PhysicsTools.PatAlgos.tools.metTools import *
addTcMET(process, 'TC')
addPfMET(process, 'PF')

## uncomment the following line to add different jet collections
## to the event content
from PhysicsTools.PatAlgos.tools.jetTools import *


## uncomment the following lines to add ak5JPTJets to your PAT output
#addJetCollection(process,cms.InputTag('JetPlusTrackZSPCorJetAntiKt5'),
#                 'AK5', 'JPT',
#                 doJTA        = True,
#                 doBTagging   = True,
#                 jetCorrLabel = ('AK5JPT', cms.vstring(['L1FastJet', 'L2Relative', 'L3Absolute'])),
#                 doType1MET   = False,
#                 doL1Cleaning = False,
#                 doL1Counters = True,
#                 genJetCollection = cms.InputTag("ak5GenJets"),
#                 doJetID      = True,
#                 jetIdLabel   = "ak5"
#                 )

## uncomment the following lines to add ak7CaloJets to your PAT output
addJetCollection(process,cms.InputTag('ak7CaloJets'),
                 'AK7', 'Calo',
                 doJTA        = True,
                 doBTagging   = False,
                 jetCorrLabel = ('AK7Calo', cms.vstring(['L1FastJet', 'L2Relative', 'L3Absolute'])),
                 doType1MET   = True,
                 doL1Cleaning = True,
                 doL1Counters = False,
                 genJetCollection=cms.InputTag("ak7GenJets"),
                 doJetID      = True,
                 jetIdLabel   = "ak7"
                 )

### uncomment the following lines to add kt4CaloJets to your PAT output
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

### uncomment the following lines to add kt6CaloJets to your PAT output
#addJetCollection(process,cms.InputTag('kt6PFJets'),
#                 'KT6', 'PF',
#                 doJTA        = True,
#                 doBTagging   = False,
#                 #jetCorrLabel = ('KT6PF', cms.vstring()), # currently not available
#                 doType1MET   = False,
#                 doL1Cleaning = True,
#                 doL1Counters = False,
#                 genJetCollection=cms.InputTag("kt6GenJets"),
#                 doJetID      = True,
#                 jetIdLabel   = "kt6"
#                 )

## uncomment the following lines to add ak5PFJets to your PAT output
switchJetCollection(process,cms.InputTag('ak5PFJets'),
                 doJTA        = True,
                 doBTagging   = True,
                 jetCorrLabel = ('AK5PF', cms.vstring(['L1FastJet', 'L2Relative', 'L3Absolute'])),
                 doType1MET   = True,
                 genJetCollection=cms.InputTag("ak5GenJets"),
                 doJetID      = True
                 )

## let it run
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
## switch to RECO input
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
