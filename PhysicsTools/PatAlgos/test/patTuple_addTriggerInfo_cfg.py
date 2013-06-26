### Set up PAT

# Import skeleton process
from PhysicsTools.PatAlgos.patTemplate_cfg import *

# Switch on "unscheduled" mode
process.options.allowUnscheduled = cms.untracked.bool( True )
#process.Tracer = cms.Service( "Tracer" )

# Load default PAT
process.load( "PhysicsTools.PatAlgos.producersLayer1.patCandidates_cff" )
process.load( "PhysicsTools.PatAlgos.selectionLayer1.selectedPatCandidates_cff" )
process.p = cms.Path(
    process.selectedPatCandidates
    )


### Get PAT trigger tools
from PhysicsTools.PatAlgos.tools.trigTools import *

# ------------------------------------------------------------------------------
# Depending on the purpose, comment/uncomment the following sections
# ------------------------------------------------------------------------------

# Add full trigger information
switchOnTrigger( process )

# # Add full example trigger matching information
# switchOnTriggerMatching( process )

# # Add stand-alone trigger information
# switchOnTriggerStandAlone( process )

# Add stand-alone example trigger matching information
# switchOnTriggerMatchingStandAlone( process )

# # Add embedded example trigger matching information
switchOnTriggerMatchEmbedding( process )


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
