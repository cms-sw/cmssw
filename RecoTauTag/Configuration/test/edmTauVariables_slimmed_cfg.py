## import skeleton process
from PhysicsTools.PatAlgos.patTemplate_cfg import cms, process
## switch to uncheduled mode
process.options.allowUnscheduled = cms.untracked.bool(True)
#process.Tracer = cms.Service("Tracer")

process.load("PhysicsTools.PatAlgos.producersLayer1.patCandidates_cff")
process.load("PhysicsTools.PatAlgos.selectionLayer1.selectedPatCandidates_cff")

process.load("SimGeneral.HepPDTESSource.pythiapdt_cfi")
process.load("RecoVertex.AdaptiveVertexFinder.inclusiveVertexing_cff")

process.load("PhysicsTools.PatAlgos.slimming.slimming_cff")
from PhysicsTools.PatAlgos.slimming.miniAOD_tools import miniAOD_customizeCommon, miniAOD_customizeMC
miniAOD_customizeCommon(process)
miniAOD_customizeMC(process)

## add command line parsing for cmsRun
## ussage is:
## cmsRun RecoTauTag/Configuration/test/edmTauVariables_slimmed_cfg.py inputFiles_load==RecoTauTag/Configuration/test/ZTT-validation.py \
## outputFile=MyOutputFile.root maxEvents=100
##
## get more inof here: https://twiki.cern.ch/twiki/bin/view/CMSPublic/SWGuideAboutPythonConfigFile#Passing_Command_Line_Arguments_T
from FWCore.ParameterSet.VarParsing import VarParsing

## define default values
from PhysicsTools.PatAlgos.patInputFiles_cff import filesRelValTTbarPileUpGENSIMRECO
process.source.fileNames = filesRelValTTbarPileUpGENSIMRECO

options = VarParsing ('analysis')
options.inputFiles = filesRelValTTbarPileUpGENSIMRECO
options.outputFile = 'patMiniAOD_standard.root'
options.maxEvents  = 100
## parse arguments
options.parseArguments()
options.inputFiles.pop(0)
process.source.fileNames = options.inputFiles 
process.out.fileName     = options.outputFile
process.maxEvents.input  = options.maxEvents

## ------------------------------------------------------
#  In addition you usually want to change the following
#  parameters:
## ------------------------------------------------------
#
#   process.GlobalTag.globaltag =  ...    ##  (according to https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideFrontierConditions)
#                                         ##
#from PhysicsTools.PatAlgos.patInputFiles_cff import filesRelValTTbarPileUpGENSIMRECO
#process.source.fileNames = filesRelValTTbarPileUpGENSIMRECO

#                                         ##
#process.maxEvents.input = 100
#                                         ##
process.out.outputCommands = process.MicroEventContentMC.outputCommands
from PhysicsTools.PatAlgos.slimming.miniAOD_tools import miniAOD_customizeOutput
miniAOD_customizeOutput(process.out)
#                                         ##
#   process.options.wantSummary = False   ##  (to suppress the long output at the end of the job)
#                                         ##
#process.out.fileName = 'patMiniAOD_standard.root'
#

