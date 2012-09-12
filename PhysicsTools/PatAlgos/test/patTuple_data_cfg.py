## Skeleton
from PhysicsTools.PatAlgos.patTemplate_cfg import *

## Options
process.options.allowUnscheduled = cms.untracked.bool( True )

## Messaging
#process.Tracer = cms.Service("Tracer")

## Conditions, In-/Output
from HLTrigger.Configuration.AutoCondGlobalTag import AutoCondGlobalTag
process.GlobalTag = AutoCondGlobalTag( process.GlobalTag, 'auto:com10' )

from PhysicsTools.PatAlgos.patInputFiles_cff import filesSingleMuRECO

## Processing
process.load( "PhysicsTools.PatAlgos.producersLayer1.patCandidates_cff" )
# for data:
# FIXME: very (too) simple to replace functionality from removed coreTools.py
process.patElectrons.addGenMatch  = False
process.patJets.addGenPartonMatch = False
process.patJets.addGenJetMatch    = False
process.patMETs.addGenMET         = False
process.patMuons.addGenMatch      = False
process.patPhotons.addGenMatch    = False
process.patTaus.addGenMatch       = False
process.patTaus.addGenJetMatch    = False
process.patJetCorrFactors.levels += [ 'L2L3Residual' ]
process.load( "PhysicsTools.PatAlgos.selectionLayer1.selectedPatCandidates_cff" )
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
process.source.fileNames = filesSingleMuRECO
#                                         ##
process.maxEvents.input = 100
#                                         ##
process.out.outputCommands += [ 'drop recoGenJets_*_*_*' ]
#                                         ##
process.out.fileName = 'patTuple_data.root'
#                                         ##
#   process.options.wantSummary = False   ##  (to suppress the long output at the end of the job)

