## import skeleton process
from PhysicsTools.PatAlgos.patTemplate_cfg import *

# load the PAT config
process.load("PhysicsTools.PatAlgos.patSequences_cff")


# Configure PAT to use PF2PAT instead of AOD sources
# this function will modify the PAT sequences.
from PhysicsTools.PatAlgos.tools.pfTools import *

postfix = "PFTypeI"
jetAlgo="AK5"
usePF2PAT(process,runPF2PAT=True, jetAlgo=jetAlgo, runOnMC=True, postfix=postfix, typeIMetCorrections=True)

## There are 3 corrections one can apply to the MET object, type-0, type-1 and type-2
## your final MET object can be type-1, type-0+1, type-1+2, or type-0+1+2
## a combination of the following instructions will allow you to get the type of MET that you desire in your
## analysis

##to add type-0 corrections to your type-1 corrected MET uncomment the following:
# getattr(process,'patType1CorrectedPFMet'+postfix).srcType1Corrections = cms.VInputTag(
#     cms.InputTag("patPFJetMETtype1p2Corr"+postfix,"type1"),
#     cms.InputTag("patPFMETtype0Corr"+postfix),
#     )
# getattr(process,'patType1p2CorrectedPFMet'+postfix).srcType1Corrections = cms.VInputTag(
#     cms.InputTag("patPFJetMETtype1p2Corr"+postfix,"type1"),
#     cms.InputTag("patPFMETtype0Corr"+postfix),
#     )
## to add type-2 corrections to your type-1 or type-0+1 corrected MET uncomment the following:
# getattr(process,'patMETs'+postfix).metSource = 'patType1p2CorrectedPFMet'+postfix

# to run second PF2PAT+PAT with different postfix uncomment the following lines
# and add the corresponding sequence to the path
#postfix2 = "PFlow2"
#jetAlgo2="AK7"
#usePF2PAT(process,runPF2PAT=True, jetAlgo=jetAlgo2, runOnMC=True, postfix=postfix2, typeIMetCorrections=True)

# to use tau-cleaned jet collection uncomment the following:
#getattr(process,"pfNoTau"+postfix).enable = True

# to switch default tau (HPS) to old default tau (shrinking cone) uncomment
# the following:
# note: in current default taus are not preselected i.e. you have to apply
# selection yourself at analysis level!
#adaptPFTaus(process,"shrinkingConePFTau",postfix=postfix)

# to use GsfElectrons instead of PF electrons
# useGsfElectrons(process,postfix)

# Let it run
process.p = cms.Path(
#    process.patDefaultSequence  +
    getattr(process,"patPF2PATSequence"+postfix)
#    second PF2PAT
#    + getattr(process,"patPF2PATSequence"+postfix2)
)

# Add PF2PAT output to the created file
from PhysicsTools.PatAlgos.patEventContent_cff import patEventContentNoCleaning
#process.load("CommonTools.ParticleFlow.PF2PAT_EventContent_cff")
#process.out.outputCommands =  cms.untracked.vstring('drop *')
process.out.outputCommands = cms.untracked.vstring('drop *',
                                                   'keep recoPFCandidates_particleFlow_*_*',
                                                   *patEventContentNoCleaning )


# top projections in PF2PAT:
getattr(process,"pfNoPileUp"+postfix).enable = True
getattr(process,"pfNoMuon"+postfix).enable = True
getattr(process,"pfNoElectron"+postfix).enable = True
getattr(process,"pfNoTau"+postfix).enable = False
getattr(process,"pfNoJet"+postfix).enable = True

# verbose flags for the PF2PAT modules
getattr(process,"pfNoMuon"+postfix).verbose = False

# enable delta beta correction for muon selection in PF2PAT?
getattr(process,"pfIsolatedMuons"+postfix).doDeltaBetaCorrection = False

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
process.out.fileName = 'patTuple_PF2PATTypeI.root'
#                                         ##
#   process.options.wantSummary = False   ##  (to suppress the long output at the end of the job)
