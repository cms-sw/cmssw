# Set of input tags for L1Extra
#
# V.M. Ghete 2010-03-14

import FWCore.ParameterSet.Config as cms

L1ExtraInputTagSetStage1 = cms.PSet(
    L1ExtraInputTags=cms.PSet(

        TagL1ExtraMuon=cms.InputTag("dqmL1ExtraParticlesStage1"),
    
        TagL1ExtraIsoEG=cms.InputTag("dqmL1ExtraParticlesStage1", "Isolated"),
        TagL1ExtraNoIsoEG=cms.InputTag("dqmL1ExtraParticlesStage1", "NonIsolated"),
    
        TagL1ExtraCenJet=cms.InputTag("dqmL1ExtraParticlesStage1", "Central"),
        TagL1ExtraForJet=cms.InputTag("dqmL1ExtraParticlesStage1", "Forward"),
        TagL1ExtraTauJet=cms.InputTag("dqmL1ExtraParticlesStage1", "Tau"),
    
        TagL1ExtraEtMissMET=cms.InputTag("dqmL1ExtraParticlesStage1", "MET"),
        TagL1ExtraEtMissHTM=cms.InputTag("dqmL1ExtraParticlesStage1", "MHT"),
    
        TagL1ExtraHFRings=cms.InputTag("dqmL1ExtraParticlesStage1")
        )
    )

