## import skeleton process
from PhysicsTools.PatAlgos.patTemplate_cfg import *

# verbose flags for the PF2PAT modules
process.options.allowUnscheduled = cms.untracked.bool(True)
#process.Tracer = cms.Service("Tracer")

runOnMC = True

if runOnMC:
    from PhysicsTools.PatAlgos.patInputFiles_cff import filesRelValProdTTbarAODSIM
    process.source.fileNames = filesRelValProdTTbarAODSIM
else:
    from PhysicsTools.PatAlgos.patInputFiles_cff import filesSingleMuRECO
    process.source.fileNames = filesSingleMuRECO
    process.GlobalTag.globaltag = cms.string( autoCond[ 'com10' ] )

# load the PAT config (for the PAT only part)
process.load("PhysicsTools.PatAlgos.producersLayer1.patCandidates_cff")
process.load("PhysicsTools.PatAlgos.selectionLayer1.selectedPatCandidates_cff")

# An empty postfix means that only PF2PAT is run,
# otherwise both standard PAT and PF2PAT are run. In the latter case PF2PAT
# collections have standard names + postfix (e.g. patElectronPFlow)
postfix = "PFlow"
jetAlgo = "AK4"
#Define Objects to be excluded from Top Projection. Default is Tau, so objects are not cleaned for taus
excludeFromTopProjection=['Tau']

# Configure PAT to use PF2PAT instead of AOD sources
# this function will modify the PAT sequences.
from PhysicsTools.PatAlgos.tools.pfTools import *
usePF2PAT(process,runPF2PAT=True, jetAlgo=jetAlgo, runOnMC=runOnMC, postfix=postfix,excludeFromTopProjection=excludeFromTopProjection)
# to run second PF2PAT+PAT with different postfix uncomment the following lines
# and add the corresponding sequence to path
#postfix2 = "PFlow2"
#jetAlgo2="AK7"
#usePF2PAT(process,runPF2PAT=True, jetAlgo=jetAlgo2, runOnMC=True, postfix=postfix2)

# to use tau-cleaned jet collection uncomment the following:
#getattr(process,"pfNoTau"+postfix).enable = True

# to switch default tau (HPS) to old default tau (shrinking cone) uncomment
# the following:
# note: in current default taus are not preselected i.e. you have to apply
# selection yourself at analysis level!
#adaptPFTaus(process,"shrinkingConePFTau",postfix=postfix)


if not runOnMC:
    # removing MC matching for standard PAT sequence
    # for the PF2PAT+PAT sequence, it is done in the usePF2PAT function
    removeMCMatchingPF2PAT( process, '' )

# Add PF2PAT output to the created file
from PhysicsTools.PatAlgos.patEventContent_cff import patEventContentNoCleaning
process.out.outputCommands = cms.untracked.vstring('drop *',
                                                   'keep recoPFCandidates_particleFlow_*_*',
						   'keep *_selectedPatJets*_*_*',
						   'drop *_selectedPatJets*%s*_caloTowers_*'%(postfix),
						   'drop *_selectedPatJets_pfCandidates_*',
						   'keep *_selectedPatElectrons*_*_*',
						   'keep *_selectedPatMuons*_*_*',
						   'keep *_selectedPatTaus*_*_*',
						   'keep *_patMETs*_*_*',
						   'keep *_selectedPatPhotons*_*_*',
						   'keep *_selectedPatTaus*_*_*',
                                                   )

## ------------------------------------------------------
#  In addition you usually want to change the following
#  parameters:
## ------------------------------------------------------
#
#   process.GlobalTag.globaltag =  ...    ##  (according to https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideFrontierConditions)
#                                         ##
#   process.source.fileNames =  ...
#
process.maxEvents.input = 10
#                                         ##
#   process.out.outputCommands = [ ... ]  ##  (e.g. taken from PhysicsTools/PatAlgos/python/patEventContent_cff.py)
#                                         ##
process.out.fileName = 'patTuple_PATandPF2PAT.root'
#                                         ##
#   process.options.wantSummary = False   ##  (to suppress the long output at the end of the job)

#process.prune(verbose=True)
