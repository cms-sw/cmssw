## import skeleton process
from PhysicsTools.PatAlgos.patTemplate_cfg import cms, process, patAlgosToolsTask
#process.Tracer = cms.Service("Tracer")

process.load("PhysicsTools.PatAlgos.producersLayer1.patCandidates_cff")
patAlgosToolsTask.add(process.patCandidatesTask)

process.load("PhysicsTools.PatAlgos.selectionLayer1.selectedPatCandidates_cff")
patAlgosToolsTask.add(process.selectedPatCandidatesTask)

## ------------------------------------------------------
#  In addition you usually want to change the following
#  parameters:
## ------------------------------------------------------
#
#   process.GlobalTag.globaltag =  ...    ##  (according to https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideFrontierConditions)
#                                         ##
process.source.fileNames = [
	'/store/relval/CMSSW_7_1_0_pre6/JetHT/RECO/PRE_R_71_V2_RelVal_jet2012C-v1/00000/2CBD40F5-E2C7-E311-8206-003048678AC0.root',
	'/store/relval/CMSSW_7_1_0_pre6/JetHT/RECO/PRE_R_71_V2_RelVal_jet2012C-v1/00000/A4DAA3A4-E0C7-E311-A427-00304867BFBC.root',
	'/store/relval/CMSSW_7_1_0_pre6/JetHT/RECO/PRE_R_71_V2_RelVal_jet2012C-v1/00000/E86E08F2-DDC7-E311-8EAC-0025905A6090.root',
]

#                                         ##
process.maxEvents.input = 10000

process.load("SimGeneral.HepPDTESSource.pythiapdt_cfi")

process.load("PhysicsTools.PatAlgos.slimming.slimming_cff")
patAlgosToolsTask.add(process.slimmingTask)

process.load("RecoVertex.AdaptiveVertexFinder.inclusiveVertexing_cff")
patAlgosToolsTask.add(process.inclusiveVertexingTask)
patAlgosToolsTask.add(process.inclusiveCandidateVertexingTask)
patAlgosToolsTask.add(process.inclusiveCandidateVertexingCvsLTask)

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
