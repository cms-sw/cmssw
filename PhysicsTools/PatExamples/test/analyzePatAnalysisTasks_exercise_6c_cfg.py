### Importing skeleton ###
import sys
import os.path
sys.path.append(os.path.abspath(os.path.expandvars(os.path.join('$CMSSW_BASE','src/PhysicsTools/PatExamples/test/'))))

from analyzePatAnalysisTasks_exercise_6c_skeleton_cfg import *

#########################
process.jecAnalyzer = cms.EDAnalyzer("WrappedEDAnalysisTasksAnalyzerJEC",
  Jets = cms.InputTag("cleanPatJets"),
  jecLevel=cms.string("L3Absolut"),
  patJetCorrFactors= cms.string('CorrFactors'),
  help=cms.bool(False),
  outputFileName=cms.string("jecAnalyzerOutput")
)

from PhysicsTools.PatUtils.tools.metUncertaintyTools import runMEtUncertainties

runMEtUncertainties(process,electronCollection="selectedPatElectrons",photonCollection=None,muonCollection="selectedPatMuons",tauCollection="selectedPatTaus",jetCollection="selectedPatJets")

process.jecAnalyzerEnUp=process.jecAnalyzer.clone(Jets = cms.InputTag("shiftedPatJetsEnDownForCorrMEt"))

process.p += process.metUncertaintySequence 
process.p += process.jecAnalyzerEnUp 

