import FWCore.ParameterSet.Config as cms

# Smoke test of the HGCAL DQM clients: construct, book from the release test
# electronics mapping, and run analyze on events without HGCAL products.
process = cms.Process("TEST")

process.source = cms.Source("EmptySource")
process.maxEvents = cms.untracked.PSet(input=cms.untracked.int32(10))

from Geometry.HGCalMapping.hgcalmapping_cff import customise_hgcalmapper
process = customise_hgcalmapper(process)
# The dense-index producers depend on CaloGeometryRecord.
process.load("Configuration.Geometry.GeometryExtendedRun4D104Reco_cff")

process.DQMStore = cms.Service("DQMStore")

process.load("DQM.HGCAL.hgcalDQM_cff")

process.p = cms.Path(process.hgcalDQMSources + process.hgcalRecoDQMSources)
