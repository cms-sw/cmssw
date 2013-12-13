# This is an example PAT configuration showing the usage of PAT on full sim samples

# Starting with a skeleton process which gets imported with the following line
from PhysicsTools.PatAlgos.patTemplate_cfg import *

# ----------------------------------------------------
# EXAMPLE 1: change the pat jet collection in the 
#            event content
# ----------------------------------------------------
#from PhysicsTools.PatAlgos.tools.jetTools import *
#switchJetCollection(process,cms.InputTag('ak4PFJets'),
#                 doJTA        = True,
#                 doBTagging   = True,
#                 jetCorrLabel = ('AK4PF', cms.vstring(['L2Relative', 'L3Absolute', 'L2L3Residual'])),
#                 doType1MET   = True,
#                 genJetCollection=cms.InputTag("ak4GenJets"),
#                 doJetID      = True
#                 )

# ----------------------------------------------------
# EXAMPLE 2: add more jet collections to the pat
#            event content
# ----------------------------------------------------
#from PhysicsTools.PatAlgos.tools.jetTools import *
#addJetCollection(process,cms.InputTag('ak8CaloJets'),
#                 'AK8', 'Calo',
#                 doJTA        = True,
#                 doBTagging   = False,
#                 jetCorrLabel = ('AK8Calo', cms.vstring(['L2Relative', 'L3Absolute'])),
#                 doType1MET   = True,
#                 doL1Cleaning = True,                 
#                 doL1Counters = False,
#                 genJetCollection=cms.InputTag("ak8GenJets"),
#                 doJetID      = True,
#                 jetIdLabel   = "ak8"
#                 )
#addJetCollection(process,cms.InputTag('ak4CaloJets'),
#                 'AK4', 'Calo',
#                 doJTA        = True,
#                 doBTagging   = True,
#                 jetCorrLabel = ('AK4Calo', cms.vstring(['L2Relative', 'L3Absolute'])),
#                 doType1MET   = True,
#                 doL1Cleaning = True,                 
#                 doL1Counters = False,
#                 genJetCollection=cms.InputTag("ak4GenJets"),
#                 doJetID      = False
#                 )

# ----------------------------------------------------
# EXAMPLE 3: add different kinds of MET to the event
#            content
# ----------------------------------------------------
#from PhysicsTools.PatAlgos.tools.metTools import *
#addTcMET(process, 'TC')
#addPfMET(process, 'PF')

# ----------------------------------------------------
# EXAMPLE 4: switch to different standard ouputs of
#            the pat tuple
# ----------------------------------------------------
## switched from cleanPatCandidates to selectedPatCandidates
#from PhysicsTools.PatAlgos.tools.coreTools import removeCleaning
#removeCleaning(process)

## add AODExtras to the event content
#from PhysicsTools.PatAlgos.patEventContent_cff import patExtraAodEventContent
#process.out.outputCommands+= patExtraAodEventContent 

# let it run
process.p = cms.Path(
    process.patDefaultSequence
    )

# ----------------------------------------------------
# You might want to change some of these default
# parameters
# ----------------------------------------------------
#process.GlobalTag.globaltag =  ...    ##  (according to https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideFrontierConditions)
#process.source.fileNames = [
#'/store/relval/CMSSW_3_1_1/RelValCosmics/GEN-SIM-RECO/STARTUP31X_V1-v2/0002/7625DA7D-E36B-DE11-865A-000423D174FE.root'
#                            ]         ##  (e.g. 'file:AOD.root')
#process.maxEvents.input = ...         ##  (e.g. -1 to run on all events)
#process.out.outputCommands = [ ... ]  ##  (e.g. taken from PhysicsTools/PatAlgos/python/patEventContent_cff.py)
#process.out.fileName = ...            ##  (e.g. 'myTuple.root')
#process.options.wantSummary = True    ##  (to suppress the long output at the end of the job)    
