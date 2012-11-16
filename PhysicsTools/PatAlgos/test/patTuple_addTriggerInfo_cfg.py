## import skeleton process
from PhysicsTools.PatAlgos.patTemplate_cfg import *

## let it run
process.p = cms.Path(
    process.patDefaultSequence
)

## add trigger information to the configuration
from PhysicsTools.PatAlgos.tools.trigTools import *
switchOnTrigger( process )

## ------------------------------------------------------
#  In addition you usually want to change the following
#  parameters:
## ------------------------------------------------------
#
#   process.GlobalTag.globaltag =  ...    ##  (according to https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideFrontierConditions)
#                                         ##
#   process.source.fileNames =  ...       ##  (e.g. 'file:AOD.root')
#                                         ##
process.maxEvents.input = 10
#                                         ##
#   process.out.outputCommands = [ ... ]  ##  (e.g. taken from PhysicsTools/PatAlgos/python/patEventContent_cff.py)
#                                         ##
process.out.fileName = 'patTuple_addTriggerInfo.root'
#                                         ##
#   process.options.wantSummary = False   ##  (to suppress the long output at the end of the job)
