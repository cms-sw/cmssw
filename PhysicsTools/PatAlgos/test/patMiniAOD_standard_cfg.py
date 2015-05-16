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

from Configuration.AlCa.GlobalTag_condDBv2 import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'MCRUN2_75_V1', '')

## ------------------------------------------------------
#  In addition you usually want to change the following
#  parameters:
## ------------------------------------------------------
#
#   process.GlobalTag.globaltag =  ...    ##  (according to https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideFrontierConditions)
#                                         ##
from PhysicsTools.PatAlgos.patInputFiles_cff import filesRelValTTbarPileUpGENSIMRECO
#process.source.fileNames = filesRelValTTbarPileUpGENSIMRECO
#process.source.fileNames = ["root://eoscms//eos/cms/store/relval/CMSSW_7_5_0_pre4/RelValTTbar_13/GEN-SIM-RECO/MCRUN2_75_V1-v1/00000/469C34DB-12F6-E411-B012-0025905B855C.root"]
process.source.fileNames = ["root://eoscms//eos/cms/store/relval/CMSSW_7_5_0_pre4/RelValTTbar_13/GEN-SIM-RECO/PU50ns_MCRUN2_75_V0-v1/00000/02B0710B-EDF7-E411-8480-0025905A607E.root"]

#                                         ##
process.maxEvents.input = 100
#                                         ##
process.out.outputCommands = process.MicroEventContentMC.outputCommands
from PhysicsTools.PatAlgos.slimming.miniAOD_tools import miniAOD_customizeOutput
miniAOD_customizeOutput(process.out)
#                                         ##
process.options.wantSummary = False   ##  (to suppress the long output at the end of the job)
#                                         ##
process.out.fileName = 'patMiniAOD_standard.root'
#

process.load("JetMETCorrections.Type1MET.correctionTermsPfMetType1Type2_cff")
#process.load("JetMETCorrections.Type1MET.correctionTermsPfMetType0PFCandidate_cff")
#process.load("JetMETCorrections.Type1MET.correctionTermsPfMetType0RecoTrack_cff")
process.load("JetMETCorrections.Type1MET.correctedMet_cff")
#process.out.outputCommands.extend([ 'keep *_correctionTermsPfMetType1Type2_*_*',
#                                    'keep *_pfMetT1_*_*',
#                                    'keep *_pfMet_*_*' ])


#process.corrPfMetType1.src = cms.InputTag('ak4PFJets')
#process.corrPfMetType1.offsetCorrLabel = cms.InputTag("ak4PFL1FastjetCorrector")
#process.corrPfMetType1.jetCorrLabel = cms.InputTag("ak4PFL1FastL2L3Corrector")

#process.patJetCorrFactorsAK4PFForMetUnc.levels = cms.vstring(
#    'L2Relative', 
#    'L3Absolute')
