import FWCore.ParameterSet.Config as cms

from DQMOffline.Trigger.ParticleNetAK4BTagClient_cfi import particleNetAK4BTagEfficiency
from DQMOffline.Trigger.ParticleNetAK8HbbTagClient_cfi import particleNetAK8HbbTagEfficiency

particleNetClientHLT = cms.Sequence(
    particleNetAK4BTagEfficiency
  + particleNetAK8HbbTagEfficiency
)

# empty particleNetClientHLT sequence when using the pp_on_AA processModifier:
#  see DQMOffline/Trigger/python/ParticleNetJetTagMonitoring_cff.py
from Configuration.ProcessModifiers.pp_on_AA_cff import pp_on_AA
pp_on_AA.toReplaceWith(particleNetClientHLT, cms.Sequence())
