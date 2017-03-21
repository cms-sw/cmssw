## import skeleton process
from PhysicsTools.PatAlgos.patTemplate_cfg import *

#process.Tracer = cms.Service("Tracer")

## load tau sequences up to selectedPatTaus
process.load("PhysicsTools.PatAlgos.producersLayer1.tauProducer_cff")
patAlgosToolsTask.add(process.makePatTausTask)
process.load("PhysicsTools.PatAlgos.selectionLayer1.tauSelector_cfi")
patAlgosToolsTask.add(process.selectedPatTaus)

## temporary fix until we find a more sustainable solution
from RecoParticleFlow.PFProducer.pfLinker_cff import particleFlowPtrs
process.particleFlowPtrs = particleFlowPtrs
patAlgosToolsTask.add(process.particleFlowPtrs)

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
process.out.fileName = 'patTuple_onlyTaus.root'
#                                         ##
#   process.options.wantSummary = False   ##  (to suppress the long output at the end of the job)
