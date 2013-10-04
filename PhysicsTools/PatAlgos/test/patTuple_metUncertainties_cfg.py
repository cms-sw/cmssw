## import skeleton process
from PhysicsTools.PatAlgos.patTemplate_cfg import *
process.load("PhysicsTools.PatUtils.patPFMETCorrections_cff")

from PhysicsTools.PatAlgos.tools.jetTools import switchJetCollection
switchJetCollection(process,cms.InputTag('ak5PFJets'),
                 doBTagging   = False,
                 jetCorrLabel = ('AK5PF', cms.vstring(['L1FastJet', 'L2Relative', 'L3Absolute'])),
                 doType1MET   = False
                 )

## let it run
process.p = cms.Path(
  process.type0PFMEtCorrection
* process.patPFMETtype0Corr
* process.patDefaultSequence
)

# apply type I/type I + II PFMEt corrections to pat::MET object
# and estimate systematic uncertainties on MET
from PhysicsTools.PatUtils.tools.metUncertaintyTools import runMEtUncertainties
runMEtUncertainties(process)

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
