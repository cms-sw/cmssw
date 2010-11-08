## import skeleton process
from PhysicsTools.PatAlgos.patTemplate_cfg import *

# the source is already defined in patTemplate_cfg.
# overriding source and various other things
#process.load("PhysicsTools.PFCandProducer.Sources.source_ZtoEles_DBS_312_cfi")
#process.source = cms.Source("PoolSource", 
#     fileNames = cms.untracked.vstring('file:myAOD.root')
#)


# process.load("PhysicsTools.PFCandProducer.Sources.source_ZtoMus_DBS_cfi")
process.options   = cms.untracked.PSet( wantSummary = cms.untracked.bool(False))

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(1000) )
process.out.fileName = cms.untracked.string('patTuple_PF2PAT.root')

# load the PAT config
process.load("PhysicsTools.PatAlgos.patSequences_cff")


# Configure PAT to use PF2PAT instead of AOD sources
# this function will modify the PAT sequences. It is currently 
# not possible to run PF2PAT+PAT and standart PAT at the same time
from PhysicsTools.PatAlgos.tools.pfTools import *

postfix = "PFlow"
jetAlgo="AK5"
usePF2PAT(process,runPF2PAT=True, jetAlgo=jetAlgo, runOnMC=True, postfix=postfix) 
# to use tau-cleaned jet collection uncomment the following:
#useTauCleanedPFJets(process, jetAlgo=jetAlgo, postfix=postfix)

# to switch default tau to HPS tau uncomment the following:
#adaptPFTaus(process,"hpsPFTau",postfix=postfix)


# Let it run
process.p = cms.Path(
#    process.patDefaultSequence  +
    getattr(process,"patPF2PATSequence"+postfix)
)

# Add PF2PAT output to the created file
from PhysicsTools.PatAlgos.patEventContent_cff import patEventContentNoCleaning
#process.load("PhysicsTools.PFCandProducer.PF2PAT_EventContent_cff")
#process.out.outputCommands =  cms.untracked.vstring('drop *')
process.out.outputCommands = cms.untracked.vstring('drop *',
                                                   *patEventContentNoCleaning ) 


# top projections in PF2PAT:

process.pfNoPileUpPFlow.enable = True 
process.pfNoMuonPFlow.enable = False 
process.pfNoElectronPFlow.enable = True 
process.pfNoTauPFlow.enable = True 
process.pfNoJetPFlow.enable = True 

process.pfNoMuon.verbose = True

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

process.MessageLogger.cerr.FwkReport.reportEvery = 10
