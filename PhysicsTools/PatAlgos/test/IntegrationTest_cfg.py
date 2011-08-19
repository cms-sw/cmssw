## import skeleton process
from PhysicsTools.PatAlgos.patTemplate_cfg import *



## ------------------------------------------------------
#  NOTE: you can use a bunch of core tools of PAT to
#  taylor your PAT configuration; for a few examples
#  uncomment the lines below
## ------------------------------------------------------
from PhysicsTools.PatAlgos.tools.coreTools import *

## remove MC matching from the default sequence
# removeMCMatching(process, ['Muons'])

## remove certain objects from the default sequence
# removeAllPATObjectsBut(process, ['Muons'])
removeSpecificPATObjects(process, ['Taus'])



## ------------------------------------------------------
#  NOTE: you can still run PAT in the 36X version on
#  input files produced within the 35X series. This
#  implies some reconfigurations, example are given
#  below.
## ------------------------------------------------------
#from PhysicsTools.PatAlgos.tools.cmsswVersionTools import *

## uncomment this line to run on an 35X input sample
#run36xOn35xInput(process)

## uncomment the following lines to add jets from a
## 35X input sample
#addJetCollection35X(process,cms.InputTag('ak7CaloJets'),
#                 'AK7', 'Calo',
#                 doJTA        = True,
#                 doBTagging   = False,
#                 jetCorrLabel = ('AK7', 'Calo'),
#                 doType1MET   = True,
#                 doL1Cleaning = True,                 
#                 doL1Counters = False,
#                 genJetCollection=cms.InputTag("ak7GenJets"),
#                 doJetID      = True,
#                 jetIdLabel   = "ak7"
#                 )

## uncomment the following lines to switch the jet
## collection from a 35X input sample
#switchJetCollection35X(process,cms.InputTag('ak5PFJets'),
#                 doJTA        = True,
#                 doBTagging   = True,
#                 jetCorrLabel = None,
#                 doType1MET   = True,
#                 genJetCollection=cms.InputTag("ak5GenJets"),
#                 doJetID      = True
#                 )



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
#   process.source.fileNames = [          ##
#    '/store/relval/CMSSW_3_5_0_pre1/RelValTTbar/GEN-SIM-RECO/STARTUP3X_V14-v1/0006/14920B0A-0DE8-DE11-B138-002618943926.root'
#   ]                                     ##  (e.g. 'file:AOD.root')
#                                         ##
#   process.maxEvents.input = ...         ##  (e.g. -1 to run on all events)
#                                         ##
#   process.out.outputCommands = [ ... ]  ##  (e.g. taken from PhysicsTools/PatAlgos/python/patEventContent_cff.py)
#                                         ##
#   process.out.fileName = ...            ##  (e.g. 'myTuple.root')
#                                         ##
#   process.options.wantSummary = True    ##  (to suppress the long output at the end of the job)    
