# Set of input tags for L1Extra
#
# V.M. Ghete 2010-03-14

import FWCore.ParameterSet.Config as cms

L1ExtraInputTagSet = cms.PSet(
    L1ExtraInputTags=cms.PSet(

        TagL1ExtraMuon=cms.InputTag("dqmL1ExtraParticles"),
    
        TagL1ExtraIsoEG=cms.InputTag("dqmL1ExtraParticles", "Isolated"),
        TagL1ExtraNoIsoEG=cms.InputTag("dqmL1ExtraParticles", "NonIsolated"),
    
        TagL1ExtraCenJet=cms.InputTag("dqmL1ExtraParticles", "Central"),
        TagL1ExtraForJet=cms.InputTag("dqmL1ExtraParticles", "Forward"),
        TagL1ExtraTauJet=cms.InputTag("dqmL1ExtraParticles", "Tau"),
    
        TagL1ExtraEtMissMET=cms.InputTag("dqmL1ExtraParticles", "MET"),
        TagL1ExtraEtMissHTM=cms.InputTag("dqmL1ExtraParticles", "MHT"),
    
        TagL1ExtraHFRings=cms.InputTag("dqmL1ExtraParticles")
        )
    )

