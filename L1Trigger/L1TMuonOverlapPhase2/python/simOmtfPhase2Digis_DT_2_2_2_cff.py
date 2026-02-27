import FWCore.ParameterSet.Config as cms

from L1Trigger.L1TMuonOverlapPhase2.simOmtfPhase2Digis_cfi import simOmtfPhase2Digis

#these patterns were gnerated with dtRefHitMinQuality = 4
#simOmtfPhase2Digis.patternsXMLFile = cms.FileInPath("L1Trigger/L1TMuon/data/omtf_config/Patterns_ExtraplMB1nadMB2DTQualAndRFixedP_DT_2_2_t30__classProb17_recalib2.xml")

#these patterns were gnerated with dtRefHitMinQuality = 2, so some pdfs are wider, but the performance is very similar as in the Patterns_ExtraplMB1nadMB2DTQualAndRFixedP_DT_2_2_t30__classProb17_recalib2.xml
#simOmtfPhase2Digis.patternsXMLFile = cms.FileInPath("L1Trigger/L1TMuon/data/omtf_config/Patterns_ExtraplMB1nadMB2DTQualAndRFixedP_DT_2_2_2_t31__classProb17_recalib2.xml")

# the patterns t30 and t31 had rpcDropAllClustersIfMoreThanMax = 0, but also cleanStubs = 1, so then its like rpcDropAllClustersIfMoreThanMax = 1, so are OK

simOmtfPhase2Digis.dtRefHitMinQuality = cms.int32(2)
simOmtfPhase2Digis.ghostBusterType = cms.string("byRefLayerAndHitQual")