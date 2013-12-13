## import skeleton process
from PhysicsTools.PatAlgos.patTemplate_cfg import *

# load the PAT config
process.load("PhysicsTools.PatAlgos.patSequences_cff")

# Configure PAT to use PF2PAT instead of AOD sources
# this function will modify the PAT sequences. It is currently
# not possible to run PF2PAT+PAT and standart PAT at the same time
from PhysicsTools.PatAlgos.tools.pfTools import *

# An empty postfix means that only PF2PAT is run,
# otherwise both standard PAT and PF2PAT are run. In the latter case PF2PAT
# collections have standard names + postfix (e.g. patElectronPFlow)
postfix = "PFlow"
usePF2PAT(process,runPF2PAT=True, jetAlgo='AK5', runOnMC=True, postfix=postfix)

# turn to false when running on data
getattr(process, "patElectrons"+postfix).embedGenMatch = True
getattr(process, "patMuons"+postfix).embedGenMatch = True

# add user data
getattr(process, "patElectrons"+postfix).userData.userFunctions.append( 'trackIso * caloIso' )
getattr(process, "patMuons"+postfix).userData.userFunctions.append( 'trackIso * caloIso' )
process.patElectrons.userData.userFunctions.append( 'trackIso * caloIso' )
process.patMuons.userData.userFunctions.append( 'trackIso * caloIso' )

getattr(process, "patElectrons"+postfix).userData.userFunctionLabels.append( 'trackIso * caloIso' )
getattr(process, "patMuons"+postfix).userData.userFunctionLabels.append( 'trackIso * caloIso' )
process.patElectrons.userData.userFunctionLabels.append( 'trackIso * caloIso' )
process.patMuons.userData.userFunctionLabels.append( 'trackIso * caloIso' )

# Let it run
process.p = cms.Path(
    getattr(process,"patPF2PATSequence"+postfix)
)
if not postfix=="":
    process.p += process.patDefaultSequence




# Add PF2PAT output to the created file
from PhysicsTools.PatAlgos.patEventContent_cff import patEventContentNoCleaning
#process.load("CommonTools.ParticleFlow.PF2PAT_EventContent_cff")
#process.out.outputCommands =  cms.untracked.vstring('drop *')
process.out.outputCommands = cms.untracked.vstring('drop *',
                                                   *patEventContentNoCleaning )

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
process.maxEvents.input = 10
#                                         ##
#   process.out.outputCommands = [ ... ]  ##  (e.g. taken from PhysicsTools/PatAlgos/python/patEventContent_cff.py)
#                                         ##
process.out.fileName = 'patTuple_userData.root'
#                                         ##
#   process.options.wantSummary = False   ##  (to suppress the long output at the end of the job)
