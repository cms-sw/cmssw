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
process.source.fileNames = [
	'/store/relval/CMSSW_7_1_0_pre6/SingleMu/RECO/PRE_R_71_V2_RelVal_zMu2012D-v1/00000/10C0893A-3CC7-E311-9483-00248C55CC9D.root',
	'/store/relval/CMSSW_7_1_0_pre6/SingleMu/RECO/PRE_R_71_V2_RelVal_zMu2012D-v1/00000/14730258-3FC7-E311-AED2-003048679182.root',
	'/store/relval/CMSSW_7_1_0_pre6/SingleMu/RECO/PRE_R_71_V2_RelVal_zMu2012D-v1/00000/1C0E64F2-40C7-E311-81D3-003048678FA6.root',
	'/store/relval/CMSSW_7_1_0_pre6/SingleMu/RECO/PRE_R_71_V2_RelVal_zMu2012D-v1/00000/221D09C4-46C7-E311-9DF4-002590596486.root',
	'/store/relval/CMSSW_7_1_0_pre6/SingleMu/RECO/PRE_R_71_V2_RelVal_zMu2012D-v1/00000/26FCDB29-3AC7-E311-B6E2-0025905A605E.root',
	'/store/relval/CMSSW_7_1_0_pre6/SingleMu/RECO/PRE_R_71_V2_RelVal_zMu2012D-v1/00000/2866963F-3FC7-E311-9931-0025905A612C.root',
	'/store/relval/CMSSW_7_1_0_pre6/SingleMu/RECO/PRE_R_71_V2_RelVal_zMu2012D-v1/00000/28E258C0-48C7-E311-B09D-003048679248.root',
	'/store/relval/CMSSW_7_1_0_pre6/SingleMu/RECO/PRE_R_71_V2_RelVal_zMu2012D-v1/00000/2A0AEBC2-43C7-E311-A924-0025905A60E0.root',
	'/store/relval/CMSSW_7_1_0_pre6/SingleMu/RECO/PRE_R_71_V2_RelVal_zMu2012D-v1/00000/2A51AC48-47C7-E311-846C-0025905A4964.root',
	'/store/relval/CMSSW_7_1_0_pre6/SingleMu/RECO/PRE_R_71_V2_RelVal_zMu2012D-v1/00000/2AB0279B-47C7-E311-A0B8-003048FFD752.root',
]
#                                         ##
process.maxEvents.input = 10000

process.load("SimGeneral.HepPDTESSource.pythiapdt_cfi")

process.load("PhysicsTools.PatAlgos.slimming.slimming_cff")

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
