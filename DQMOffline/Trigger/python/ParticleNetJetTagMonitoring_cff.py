import FWCore.ParameterSet.Config as cms

from DQMOffline.Trigger.ParticleNetAK4BTagMonitoring_cfi import particleNetAK4BTagMonitoring
from DQMOffline.Trigger.ParticleNetAK8HbbTagMonitoring_cfi import particleNetAK8HbbTagMonitoring

particleNetMonitoringHLT = cms.Sequence(
    particleNetAK4BTagMonitoring
  + particleNetAK8HbbTagMonitoring
)

# empty particleNetMonitoringHLT sequence when using the pp_on_AA processModifier:
#  HLT-PNET DQM can trigger the execution of modules to run inference
#  on offline jet collections which are not present in HIon workflows
from Configuration.ProcessModifiers.pp_on_AA_cff import pp_on_AA
pp_on_AA.toReplaceWith(particleNetMonitoringHLT, cms.Sequence())
