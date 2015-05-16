## import skeleton process
from PhysicsTools.PatAlgos.patTemplate_cfg import *
## switch to uncheduled mode
process.options.allowUnscheduled = cms.untracked.bool(True)

process.load("PhysicsTools.PatAlgos.producersLayer1.patCandidates_cff")
process.load("PhysicsTools.PatAlgos.selectionLayer1.selectedPatCandidates_cff")
process.load("PhysicsTools.PatUtils.patPFMETCorrections_cff")

from PhysicsTools.PatAlgos.tools.jetTools import switchJetCollection
switchJetCollection(process,
                    jetSource = cms.InputTag('ak4PFJets'),
                    jetCorrections = ('AK4PF', ['L1FastJet', 'L2Relative', 'L3Absolute'], '')
                    )

# apply type I PFMEt corrections to pat::MET object
# and estimate systematic uncertainties on MET
from PhysicsTools.PatUtils.tools.runMETCorrectionsAndUncertainties import runMETCorrectionsAndUncertainties

runMETCorrectionsAndUncertainties(process, metType="PF",
                                  correctionLevel=["T1"],
                                  computeUncertainties=True,
                                  produceIntermediateCorrections=False,
                                  addToPatDefaultSequence=False,
                                  jetCollectionUnskimmed="patJets",
                                  jetCollection="selectedPatJets",
                                  postfix="",
                                  )
    

## ------------------------------------------------------
#  In addition you usually want to change the following
#  parameters:
## ------------------------------------------------------
#
#   process.GlobalTag.globaltag =  ...    ##  (according to https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideFrontierConditions)
#                                         ##
## switch to RECO input
from PhysicsTools.PatAlgos.patInputFiles_cff import filesRelValProdTTbarAODSIM
process.source.fileNames = filesRelValProdTTbarAODSIM
#                                         ##
process.maxEvents.input = 10
#                                         ##
#   process.out.outputCommands = [ ... ]  ##  (e.g. taken from PhysicsTools/PatAlgos/python/patEventContent_cff.py)
#                                         ##
process.out.fileName = 'patTuple_metUncertainties.root'
#                                         ##
#   process.options.wantSummary = False   ##  (to suppress the long output at the end of the job)
