import FWCore.ParameterSet.Config as cms

# This modifier is for running NANOAOD production from L1 and HLT steps, 
# whitout running RECO and PAT steps. 
# It excludes all GEN sequences that depend on PAT collections.

nano_l1_hlt = cms.Modifier()
