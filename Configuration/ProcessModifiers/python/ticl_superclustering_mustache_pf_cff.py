import FWCore.ParameterSet.Config as cms

# Modifier (to be applied on top of ticl_v5 modifier) to use the pre-TICLv5 superclustering code (that translates tracksters to PFClusters before running Mustache)
ticl_superclustering_mustache_pf = cms.Modifier() 
