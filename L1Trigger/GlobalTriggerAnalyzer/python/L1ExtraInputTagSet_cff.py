# Set of input tags for L1Extra in agreement with L1Reco_cff
#
# V.M. Ghete 2012-05-22

import FWCore.ParameterSet.Config as cms

L1ExtraInputTagSet = cms.PSet(
    L1ExtraInputTags=cms.PSet(

        TagL1ExtraMuon=cms.InputTag("l1extraParticles"),
    
        TagL1ExtraIsoEG=cms.InputTag("l1extraParticles", "Isolated"),
        TagL1ExtraNoIsoEG=cms.InputTag("l1extraParticles", "NonIsolated"),
    
        TagL1ExtraCenJet=cms.InputTag("l1extraParticles", "Central"),
        TagL1ExtraForJet=cms.InputTag("l1extraParticles", "Forward"),
        TagL1ExtraTauJet=cms.InputTag("l1extraParticles", "Tau"),
    
        TagL1ExtraEtMissMET=cms.InputTag("l1extraParticles", "MET"),
        TagL1ExtraEtMissHTM=cms.InputTag("l1extraParticles", "MHT"),
    
        TagL1ExtraHFRings=cms.InputTag("l1extraParticles")
        )
    )

