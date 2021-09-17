## import skeleton process
from PhysicsTools.PatAlgos.patTemplate_cfg import *

## to run in scheduled mode uncomment the following lines
#process.load("PhysicsTools.PatAlgos.patSequences_cff")
#process.p = cms.Path(
#    process.patDefaultSequence
#    )

## to run in un-scheduled mode uncomment the following lines
process.load("PhysicsTools.PatAlgos.producersLayer1.patCandidates_cff")
patAlgosToolsTask.add(process.patCandidatesTask)
# Temporary customize to the unit tests that fail due to old input samples
process.patTaus.skipMissingTauID = True
process.patMuons.addTriggerMatching = False

process.load("PhysicsTools.PatAlgos.selectionLayer1.selectedPatCandidates_cff")
patAlgosToolsTask.add(process.selectedPatCandidatesTask)

#process.Tracer = cms.Service("Tracer")
process.p = cms.Path(
    process.selectedPatCandidates
    )

process.patLowPtElectrons.addElectronID = False
process.patLowPtElectrons.electronSource = "gedGsfElectrons"
process.patLowPtElectrons.genParticleMatch = "electronMatch"
process.selectedPatLowPtElectrons.cut = "pt>99999."

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
process.out.fileName = 'IntegrationTest.root'
#                                         ##
#   process.options.wantSummary = False   ##  (to suppress the long output at the end of the job)
