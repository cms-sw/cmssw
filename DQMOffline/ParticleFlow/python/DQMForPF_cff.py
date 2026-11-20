import FWCore.ParameterSet.Config as cms

from DQMOffline.ParticleFlow.particleFlowDQM_cff import pfJetAnalyzerDQM
from DQMOffline.ParticleFlow.particleFlowDQM_cff import pfPuppiJetAnalyzerDQM
from DQMOffline.ParticleFlow.particleFlowDQM_cff import pfJetDQMPostProcessor
from DQMOffline.ParticleFlow.particleFlowDQM_cff import pfAnalyzerDQM, pfAnalyzerMiniAODDQM
from DQMOffline.ParticleFlow.offsetAnalyzerDQM_cff import offsetAnalyzerDQM, offsetAnalyzerMiniAODDQM
from DQMOffline.ParticleFlow.offsetAnalyzerDQM_cff import offsetDQMPostProcessor
# Use also other POGs' analyzers for extended checks
from Validation.RecoMET.METRelValForDQM_cff import *
from Validation.RecoJets.JetValidation_cff import *

DQMOfflinePFAnalyzer = cms.Sequence(
  pfAnalyzerDQM
)

DQMOfflinePFMiniAOD = cms.Sequence(
  pfAnalyzerMiniAODDQM +
  pfJetAnalyzerDQM +
  pfPuppiJetAnalyzerDQM +
  offsetAnalyzerMiniAODDQM
)

DQMOfflinePF = cms.Sequence(
  pfAnalyzerDQM +
  pfJetAnalyzerDQM + # using pat:Jets
  pfPuppiJetAnalyzerDQM + # using pat:Jets
  offsetAnalyzerDQM
)

DQMHarvestPF = cms.Sequence(
  pfJetDQMPostProcessor +
  offsetDQMPostProcessor
)

# MET & Jets sequence
DQMOfflinePFExtended = cms.Sequence(
    METValidationMiniAOD +
    JetValidationMiniAOD
)
