## import skeleton process
from PhysicsTools.PatAlgos.patTemplate_cfg import *

## load met sequences up to patMETs
process.load("PhysicsTools.PatAlgos.producersLayer1.metProducer_cff")

## make sure to keep the created objects
process.out.outputCommands = ['keep *_patMETs*_*_*',]

## let it run
process.p = cms.Path(
     process.makePatMETs
)

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
process.out.fileName = 'patTuple_onlyMET.root'
#                                         ##
#   process.options.wantSummary = False   ##  (to suppress the long output at the end of the job)
