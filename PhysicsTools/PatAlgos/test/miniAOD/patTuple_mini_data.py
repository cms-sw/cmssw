## import skeleton process
from PhysicsTools.PatAlgos.patTemplate_cfg import cms, process
## switch to uncheduled mode
process.options.allowUnscheduled = cms.untracked.bool(True)
#process.Tracer = cms.Service("Tracer")

process.load("PhysicsTools.PatAlgos.producersLayer1.patCandidates_cff")
process.load("PhysicsTools.PatAlgos.selectionLayer1.selectedPatCandidates_cff")

## ------------------------------------------------------
#  In addition you usually want to change the following
#  parameters:
## ------------------------------------------------------
#
#   process.GlobalTag.globaltag =  ...    ##  (according to https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideFrontierConditions)
#                                         ##
process.source.fileNames = {'/store/relval/CMSSW_7_0_0/SingleMu/RECO/GR_R_70_V1_RelVal_zMu2012D-v2/00000/0259E46E-F698-E311-8CFD-003048FF9AC6.root'}

#                                         ##
process.maxEvents.input = 10000

process.load("SimGeneral.HepPDTESSource.pythiapdt_cfi")

process.load("PhysicsTools.PatAlgos.slimming.slimming_cff")
process.load("RecoVertex.AdaptiveVertexFinder.inclusiveVertexing_cff")

process.GlobalTag.globaltag = "GR_R_70_V1::All"

from PhysicsTools.PatAlgos.slimming.miniAOD_tools import miniAOD_customizeCommon, miniAOD_customizeData
miniAOD_customizeCommon(process)
miniAOD_customizeData(process)

#                                         ##
#   process.options.wantSummary = False   ##  (to suppress the long output at the end of the job)
#                                         ##
process.out.fileName = 'patTuple_mini_singlemu.root'
process.out.outputCommands = process.MicroEventContent.outputCommands
from PhysicsTools.PatAlgos.slimming.miniAOD_tools import miniAOD_customizeOutput
miniAOD_customizeOutput(process.out)
