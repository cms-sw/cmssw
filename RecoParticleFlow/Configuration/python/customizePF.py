import FWCore.ParameterSet.Config as cms

#
# customization function for ECAL PF-rechit thresholds
# - ZS everywhere
# - 1-sigma noise equivalent from 2018
#
def ecalNoSRPFCut1sigma(process):
   import RecoParticleFlow.PFClusterProducer.particleFlowZeroSuppressionECAL_cff as _pfZS
   if hasattr(process, "particleFlowRecHitECAL"):
      process.particleFlowRecHitECAL.producers[0].srFlags = ""
      process.particleFlowRecHitECAL.producers[1].srFlags = ""
      process.particleFlowRecHitECAL.producers[0].qualityTests[0].thresholds = _pfZS._particle_flow_zero_suppression_ECAL_2018_B.thresholds
      process.particleFlowRecHitECAL.producers[1].qualityTests[0].thresholds = _pfZS._particle_flow_zero_suppression_ECAL_2018_B.thresholds
      process.particleFlowRecHitECAL.producers[0].qualityTests[0].applySelectionsToAllCrystals = cms.bool(True)
      process.particleFlowRecHitECAL.producers[1].qualityTests[0].applySelectionsToAllCrystals = cms.bool(True)
   return process

