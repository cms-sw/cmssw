## import skeleton process
from PhysicsTools.PatAlgos.patTemplate_cfg import *


process.source.fileNames = [
'/store/relval/CMSSW_5_2_4-START52_V9_special_SLC5xSLC6/RelValTTbar/GEN-SIM-RECO/v1/0000/B2C5420B-0E9E-E111-9744-003048FFD76E.root'
]

# Get a list of good primary vertices, in 42x, these are DAF vertices
from PhysicsTools.SelectorUtils.pvSelector_cfi import pvSelector
process.goodOfflinePrimaryVertices = cms.EDFilter(
    "PrimaryVertexObjectFilter",
    filterParams = pvSelector.clone( minNdof = cms.double(4.0), maxZ = cms.double(24.0) ),
    src=cms.InputTag('offlinePrimaryVertices')
    )


# Configure PAT to use PF2PAT instead of AOD sources
# this function will modify the PAT sequences. It is currently 
# not possible to run PF2PAT+PAT and standart PAT at the same time
from PhysicsTools.PatAlgos.tools.pfTools import *
postfix = "PFlow"
usePF2PAT(process,runPF2PAT=True,
          jetAlgo='AK5', runOnMC=True, postfix=postfix,
	  jetCorrections=('AK5PFchs', ['L1FastJet', 'L2Relative', 'L3Absolute']),
          pvCollection=cms.InputTag('goodOfflinePrimaryVertices')
          )
process.pfPileUpPFlow.checkClosestZVertex = False

process.patseq = cms.Sequence(    
    process.goodOfflinePrimaryVertices*
    getattr(process,"patPF2PATSequence"+postfix)
    )

# Adjust the event content
process.out.outputCommands += [
    'keep *_selectedPat*_*_*',
    'keep *_goodOfflinePrimaryVertices*_*_*',    
    'keep double_*_rho_*'
]


## let it run
process.p = cms.Path(
    process.patseq
    )

## ------------------------------------------------------
#  In addition you usually want to change the following
#  parameters:
## ------------------------------------------------------
#
#   process.GlobalTag.globaltag =  ...    ##  (according to https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideFrontierConditions)
#                                         ##
#   process.source.fileNames = [          ##
#    '/store/relval/CMSSW_3_8_6/RelValTTbar/GEN-SIM-RECO/START38_V13-v1/0065/F438C4C4-BCE7-DF11-BC6B-002618943885.root'
#   ]                                     ##  (e.g. 'file:AOD.root')
#                                         ##
#   process.maxEvents.input = ...         ##  (e.g. -1 to run on all events)
#                                         ##
#   process.out.outputCommands = [ ... ]  ##  (e.g. taken from PhysicsTools/PatAlgos/python/patEventContent_cff.py)
#                                         ##
#   process.out.fileName = ...            ##  (e.g. 'myTuple.root')
#                                         ##
process.options.wantSummary = True        ##  (to suppress the long output at the end of the job)    
