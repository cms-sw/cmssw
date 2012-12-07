## import skeleton process
from PhysicsTools.PatAlgos.patTemplate_cfg import *

## to run in scheduled mode uncomment the following lines
#process.load("PhysicsTools.PatAlgos.patSequences_cff")
#process.p = cms.Path(
#    process.patDefaultSequence
#    )

## to run in un-scheduled mode uncomment the following lines
process.load("PhysicsTools.PatAlgos.producersLayer1.patCandidates_cff")
process.load("PhysicsTools.PatAlgos.selectionLayer1.selectedPatCandidates_cff")

process.options.allowUnscheduled = cms.untracked.bool(True)
#process.Tracer = cms.Service("Tracer")
process.p = cms.Path(
    process.selectedPatCandidates
    )



## ------------------------------------------------------
#  In addition you usually want to change the following
#  parameters:
## ------------------------------------------------------
#
#   process.GlobalTag.globaltag =  ...    ##  (according to https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideFrontierConditions)
#                                         ##
#   process.source.fileNames = [          ##  (e.g. 'file:AOD.root')
#     '/store/data/Run2012B/DoubleMu/AOD/PromptReco-v1/000/193/774/0CDC3936-889B-E111-9F82-001D09F25041.root'
#    ]
#                                         ##
process.maxEvents.input = 100
#                                         ##
#   process.out.outputCommands = [ ... ]  ##  (e.g. taken from PhysicsTools/PatAlgos/python/patEventContent_cff.py)
#                                         ##
process.out.fileName = 'IntegrationTest.root'
#                                         ##
#   process.options.wantSummary = False   ##  (to suppress the long output at the end of the job)
