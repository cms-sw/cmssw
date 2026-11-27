import FWCore.ParameterSet.Config as cms

from DQMOffline.ParticleFlow.particleFlowDQM_cff import pfJetAnalyzerDQM, pfJetAnalyzerMiniAODDQM
from DQMOffline.ParticleFlow.particleFlowDQM_cff import pfPuppiJetAnalyzerDQM, pfPuppiJetAnalyzerMiniAODDQM
from DQMOffline.ParticleFlow.particleFlowDQM_cff import pfJetDQMPostProcessor, pfJetDQMPostProcessorMiniAOD
from DQMOffline.ParticleFlow.particleFlowDQM_cff import pfAnalyzerDQM, pfAnalyzerMiniAODDQM
from DQMOffline.ParticleFlow.offsetAnalyzerDQM_cff import offsetAnalyzerDQM, offsetAnalyzerMiniAODDQM
from DQMOffline.ParticleFlow.offsetAnalyzerDQM_cff import offsetDQMPostProcessor, offsetDQMPostProcessorMiniAOD
# Use also other POGs' analyzers for extended checks
from Validation.RecoMET.METRelValForDQM_cff import *
from Validation.RecoJets.JetValidation_cff import *

DQMOfflinePFAnalyzer = cms.Sequence(
  pfAnalyzerDQM
)

DQMOfflinePFMiniAOD = cms.Sequence(
  pfAnalyzerMiniAODDQM +
  pfJetAnalyzerMiniAODDQM +
  pfPuppiJetAnalyzerMiniAODDQM +
  offsetAnalyzerMiniAODDQM
)

DQMOfflinePF = cms.Sequence(
  pfAnalyzerDQM +
  pfJetAnalyzerDQM +
  pfPuppiJetAnalyzerDQM +
  offsetAnalyzerDQM
)

DQMHarvestPF = cms.Sequence(
  pfJetDQMPostProcessor +
  offsetDQMPostProcessor
)

DQMHarvestPFMiniAOD = cms.Sequence(
  pfJetDQMPostProcessorMiniAOD +
  offsetDQMPostProcessorMiniAOD
)

# MET & Jets sequence
DQMOfflinePFExtended = cms.Sequence(
    METValidationMiniAOD +
    JetValidationMiniAOD
)
