import FWCore.ParameterSet.Config as cms

process = cms.Process("READ")

process.load("Geometry.CMSCommonData.cmsIdealGeometryXML_cfi")

process.load("Geometry.EcalMapping.EcalMapping_cfi")
process.load("Geometry.EcalMapping.EcalMappingRecord_cfi")

process.load("Geometry.CaloEventSetup.EcalTrigTowerConstituents_cfi")

process.source = cms.Source("EmptyIOVSource",
                            firstValue = cms.uint64(1),
                            lastValue = cms.uint64(1),
                            timetype = cms.string('runnumber'),
                            interval = cms.uint64(1)
                            )

process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.GlobalTag.globaltag = "GR10_P_V12::All"

process.GlobalTag.toGet = cms.VPSet(
  cms.PSet(record = cms.string("EcalDQMChannelStatusRcd"),
           tag = cms.string("EcalDQMChannelStatus_v1_hlt"),
           connect = cms.untracked.string("sqlite_file:mask-ECAL.db")
          ),
  cms.PSet(record = cms.string("EcalDQMTowerStatusRcd"),
           tag = cms.string("EcalDQMTowerStatus_v1_hlt"),
           connect = cms.untracked.string("sqlite_file:mask-ECAL.db")
          )
)

process.read = cms.EDAnalyzer("EcalDQMStatusReader",
  verbose = cms.untracked.bool(True),
)

process.p = cms.Path(process.read)

