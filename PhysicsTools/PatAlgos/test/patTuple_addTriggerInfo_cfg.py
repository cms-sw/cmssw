### Set up PAT

# Import skeleton process
from PhysicsTools.PatAlgos.patTemplate_cfg import *

#process.Tracer = cms.Service( "Tracer" )

# Load default PAT
process.load( "PhysicsTools.PatAlgos.producersLayer1.patCandidates_cff" )
patAlgosToolsTask.add(process.patCandidatesTask)
#Temporary customize to the unit tests that fail due to old input samples
process.patTaus.skipMissingTauID = True

process.load( "PhysicsTools.PatAlgos.selectionLayer1.selectedPatCandidates_cff" )
patAlgosToolsTask.add(process.selectedPatCandidatesTask)

process.p = cms.Path(
    process.selectedPatCandidates
    )

process.patLowPtElectrons.addElectronID = False
process.patLowPtElectrons.electronSource = "gedGsfElectrons"
process.patLowPtElectrons.genParticleMatch = "electronMatch"
process.selectedPatLowPtElectrons.cut = "pt>99999."

process.filteredDisplacedMuons.srcMuons = "muons"
process.selectedPatDisplacedMuons.cut = "pt>99999."

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
