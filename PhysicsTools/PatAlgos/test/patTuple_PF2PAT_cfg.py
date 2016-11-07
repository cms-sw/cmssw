## import skeleton process
from PhysicsTools.PatAlgos.patTemplate_cfg import *
# verbose flags for the PF2PAT modules

#process.Tracer = cms.Service("Tracer")

# Configure PAT to use PF2PAT instead of AOD sources
# this function will modify the PAT sequences.
from PhysicsTools.PatAlgos.tools.pfTools import *
postfix = "PFlow"
jetAlgo="AK4"
usePF2PAT(process,runPF2PAT=True, jetAlgo=jetAlgo, runOnMC=True, postfix=postfix)

# to turn on type-1 MET corrections, use the following call
#usePF2PAT(process,runPF2PAT=True, jetAlgo=jetAlgo, runOnMC=True, postfix=postfix, typeIMetCorrections=True)

# to switch default tau (HPS) to old default tau (shrinking cone) uncomment
# the following:
# note: in current default taus are not preselected i.e. you have to apply
# selection yourself at analysis level!
#adaptPFTaus(process,"shrinkingConePFTau",postfix=postfix)

# Add PF2PAT output to the created file
from PhysicsTools.PatAlgos.patEventContent_cff import patEventContentNoCleaning
#process.load("CommonTools.ParticleFlow.PF2PAT_EventContent_cff")
#process.out.outputCommands =  cms.untracked.vstring('drop *')
process.out.outputCommands = cms.untracked.vstring('drop *',
                                                   'keep recoPFCandidates_particleFlow_*_*',
						   'keep *_selectedPatJets*_*_*',
						   'drop *_selectedPatJets*_caloTowers_*',
						   'keep *_selectedPatElectrons*_*_*',
						   'keep *_selectedPatMuons*_*_*',
						   'keep *_selectedPatTaus*_*_*',
						   'keep *_patMETs*_*_*',
						   'keep *_selectedPatPhotons*_*_*',
						   'keep *_selectedPatTaus*_*_*',
                                                   )


# top projections in PF2PAT:
getattr(process,"pfNoPileUpJME"+postfix).enable = True
getattr(process,"pfNoMuonJME"+postfix).enable = True
getattr(process,"pfNoElectronJME"+postfix).enable = True
getattr(process,"pfNoTau"+postfix).enable = False
getattr(process,"pfNoJet"+postfix).enable = True
# to use tau-cleaned jet collection uncomment the following:
#getattr(process,"pfNoTau"+postfix).enable = True

# verbose flags for the PF2PAT modules
getattr(process,"pfNoMuonJME"+postfix).verbose = False

# enable delta beta correction for muon selection in PF2PAT?
getattr(process,"pfIsolatedMuons"+postfix).doDeltaBetaCorrection = cms.bool(False)

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
process.out.fileName = 'patTuple_PF2PAT.root'
#                                         ##
#   process.options.wantSummary = False   ##  (to suppress the long output at the end of the job)
