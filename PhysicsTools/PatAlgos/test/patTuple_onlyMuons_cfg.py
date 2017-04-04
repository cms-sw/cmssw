## import skeleton process
from PhysicsTools.PatAlgos.patTemplate_cfg import *

#process.Tracer = cms.Service("Tracer")

## load tau sequences up to selectedPatMuons
process.load("PhysicsTools.PatAlgos.producersLayer1.muonProducer_cff")
patAlgosToolsTask.add(process.makePatMuonsTask)

process.load("PhysicsTools.PatAlgos.selectionLayer1.muonSelector_cfi")
patAlgosToolsTask.add(process.selectedPatMuons)

## make sure to keep the created objects
process.out.outputCommands = ['keep *_selectedPat*_*_*']

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
process.out.fileName = 'patTuple_onlyMuons.root'
#                                         ##
#   process.options.wantSummary = False   ##  (to suppress the long output at the end of the job)
