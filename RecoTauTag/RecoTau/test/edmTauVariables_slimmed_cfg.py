## import skeleton process
from PhysicsTools.PatAlgos.patTemplate_cfg import cms, process, patAlgosToolsTask
#process.Tracer = cms.Service("Tracer")

process.load("PhysicsTools.PatAlgos.producersLayer1.patCandidates_cff")
patAlgosToolsTask.add(process.patCandidatesTask)

process.load("PhysicsTools.PatAlgos.selectionLayer1.selectedPatCandidates_cff")
patAlgosToolsTask.add(process.selectedPatCandidatesTask)

process.load("SimGeneral.HepPDTESSource.pythiapdt_cfi")
process.load("RecoVertex.AdaptiveVertexFinder.inclusiveVertexing_cff")
patAlgosToolsTask.add(process.inclusiveVertexingTask)
patAlgosToolsTask.add(process.inclusiveCandidateVertexingTask)
patAlgosToolsTask.add(process.inclusiveCandidateVertexingCvsLTask)

process.load("PhysicsTools.PatAlgos.slimming.slimming_cff")
patAlgosToolsTask.add(process.slimmingTask)

from PhysicsTools.PatAlgos.slimming.miniAOD_tools import miniAOD_customizeCommon, miniAOD_customizeMC
miniAOD_customizeCommon(process)
miniAOD_customizeMC(process)

## ------------------------------------------------------
#  In addition you usually want to change the following
#  parameters:
## ------------------------------------------------------
#
#   process.GlobalTag.globaltag =  ...    ##  (according to https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideFrontierConditions)
#                                         ##
from PhysicsTools.PatAlgos.patInputFiles_cff import filesRelValTTbarPileUpGENSIMRECO
process.source.fileNames = filesRelValTTbarPileUpGENSIMRECO

#                                         ##
process.maxEvents.input = 100
#                                         ##
process.out.outputCommands = process.MicroEventContentMC.outputCommands
from PhysicsTools.PatAlgos.slimming.miniAOD_tools import miniAOD_customizeOutput
miniAOD_customizeOutput(process.out)
## added for mvaIsolation on miniAOD testing
process.out.outputCommands += [
  "keep *_particleFlow_*_*",
  "keep recoTracks_generalTracks_*_*",
#  "keep hpsPFTau*_*_*_*",
  "keep *_hpsPFTauProducer_*_*",
  "keep *_hpsPFTauDiscriminationByDecayModeFindingNewDMs_*_*",
  "keep *_hpsPFTauChargedIsoPtSum_*_*",
  "keep *_hpsPFTauNeutralIsoPtSum_*_*",
  "keep *_hpsPFTauPUcorrPtSum_*_*",
  "keep *_hpsPFTauPhotonPtSumOutsideSignalCone_*_*",
  "keep *_hpsPFTauFootprintCorrection_*_*",
  "keep *_hpsPFTauTransverseImpactParameters_*_*",
  ]
#                                         ##
#   process.options.wantSummary = False   ##  (to suppress the long output at the end of the job)
#                                         ##
process.out.fileName = 'patMiniAOD_standard.root'
#

