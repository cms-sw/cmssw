### Set up PAT

# Import skeleton process
from PhysicsTools.PatAlgos.patTemplate_cfg import *

# Switch on "unscheduled" mode
process.options.allowUnscheduled = cms.untracked.bool( True )
#process.Tracer = cms.Service( "Tracer" )

# Load default PAT
process.load( "PhysicsTools.PatAlgos.producersLayer1.patCandidates_cff" )
process.load( "PhysicsTools.PatAlgos.selectionLayer1.selectedPatCandidates_cff" )


### Set up PAT trigger information

process.load( "PhysicsTools.PatAlgos.triggerLayer1.triggerProducer_cfi" )
process.load( "PhysicsTools.PatAlgos.triggerLayer1.triggerEventProducer_cfi" )

# ------------------------------------------------------------------------------
# Depending on the purpose, comment/uncomment the following sections
# ------------------------------------------------------------------------------

# Add full trigger information
from PhysicsTools.PatAlgos.patEventContent_cff import patTriggerEventContent
process.out.outputCommands += patTriggerEventContent

# # Add full example trigger matching information
# process.load( "PhysicsTools.PatAlgos.triggerLayer1.triggerMatcher_cfi" )
# from PhysicsTools.PatAlgos.triggerLayer1.triggerMatcher_cfi import triggerMatchingDefaultInputTags
# process.patTriggerEvent.patTriggerMatches = triggerMatchingDefaultInputTags
# from PhysicsTools.PatAlgos.patEventContent_cff import patTriggerEventContent
# process.out.outputCommands += patTriggerEventContent

# # Add stand-alone trigger information
# from PhysicsTools.PatAlgos.patEventContent_cff import patTriggerStandAloneEventContent
# process.out.outputCommands += patTriggerStandAloneEventContent

# Add stand-alone example trigger matching information
# process.load( "PhysicsTools.PatAlgos.triggerLayer1.triggerMatcher_cfi" )
# from PhysicsTools.PatAlgos.patEventContent_cff import patTriggerStandAloneEventContent
# process.out.outputCommands += patTriggerStandAloneEventContent

# # Add embedded example trigger matching information
# process.load( "PhysicsTools.PatAlgos.triggerLayer1.triggerMatcher_cfi" )
# process.load( "PhysicsTools.PatAlgos.triggerLayer1.triggerMatchEmbedder_cfi" )
# from PhysicsTools.PatAlgos.patEventContent_cff import patEventContentTriggerMatch
# process.out.outputCommands.append( 'drop *' )
# process.out.outputCommands += patEventContentTriggerMatch


# ------------------------------------------------------------------------------
#  In addition you possibly want to change the following parameters:
# ------------------------------------------------------------------------------
#
#   process.GlobalTag.globaltag =  ...                # according to https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideFrontierConditions
#                                                     #
from PhysicsTools.PatAlgos.patInputFiles_cff import filesRelValProdTTbarAODSIM
process.source.fileNames = filesRelValProdTTbarAODSIM
#                                                     #
process.maxEvents.input = 10                          # number of events to process
#                                                     #
process.out.fileName = 'patTuple_addTriggerInfo.root' # name of the input EDM file
#                                                     #
#   process.options.wantSummary = False               # suppresses the long output at the end of the job
