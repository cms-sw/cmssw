import FWCore.ParameterSet.Config as cms

# This modifier enables the usage of MTD into the Phase2 HLT
# This includes:
# - MTD local reconstruction
# - fast timing global reconstruction (TrackExtenderWithMTD)
# - 4D vertexing
mtd_at_hlt = cms.Modifier()
