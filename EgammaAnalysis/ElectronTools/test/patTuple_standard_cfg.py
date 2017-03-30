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

process.load("PhysicsTools.PatAlgos.selectionLayer1.selectedPatCandidates_cff")

#process.Tracer = cms.Service("Tracer")
process.p = cms.Path(
    process.selectedPatCandidates
    )

process.out.outputCommands.append( 'keep *_offlinePrimaryVertices_*_*' )
process.out.outputCommands.append( 'keep double_kt6PFJets_rho_RECO' )

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
process.out.fileName = 'patTuple_standard.root'
#                                         ##
#   process.options.wantSummary = False   ##  (to suppress the long output at the end of the job)

