import FWCore.ParameterSet.Config as cms

process = cms.Process("DB")

process.load("Geometry.CMSCommonData.cmsIdealGeometryXML_cfi")

process.load("Geometry.EcalMapping.EcalMapping_cfi")
process.load("Geometry.EcalMapping.EcalMappingRecord_cfi")

process.load("Geometry.CaloEventSetup.CaloGeometry_cfi")
process.load("Geometry.CaloEventSetup.CaloTopology_cfi")
process.load("Geometry.CaloEventSetup.EcalTrigTowerConstituents_cfi")

process.source = cms.Source("EmptyIOVSource",
                            firstValue = cms.uint64(1),
                            lastValue = cms.uint64(1),
                            timetype = cms.string('runnumber'),
                            interval = cms.uint64(1)
                            )

process.load("DQM.Integration.test.FrontierCondition_GT_cfi")

process.GlobalTag.toGet = cms.VPSet(
  cms.PSet(record = cms.string("EcalDQMChannelStatusRcd"),
#           tag = cms.string("EcalDQMChannelStatus_v1_hlt"),
#           tag = cms.string("EcalDQMChannelStatus_v1_express"),
#           tag = cms.string("EcalDQMChannelStatus_v1_offline"),           
#           connect = cms.untracked.string("sqlite_file:mask-ECAL.db")
#           connect = cms.untracked.string("frontier://(proxyurl=http://localhost:3128)(serverurl=http://localhost:8000/FrontierOnProd)(serverurl=http://localhost:8000/FrontierOnProd)(retrieve-ziplevel=0)(failovertoserver=no)/CMS_COND_34X_ECAL")
          ),
  cms.PSet(record = cms.string("EcalDQMTowerStatusRcd"),
#           tag = cms.string("EcalDQMTowerStatus_v1_hlt"),
#           tag = cms.string("EcalDQMTowerStatus_v1_express"),
#           tag = cms.string("EcalDQMTowerStatus_v1_offline"),
#           connect = cms.untracked.string("sqlite_file:mask-ECAL.db")
#           connect = cms.untracked.string("frontier://(proxyurl=http://localhost:3128)(serverurl=http://localhost:8000/FrontierOnProd)(serverurl=http://localhost:8000/FrontierOnProd)(retrieve-ziplevel=0)(failovertoserver=no)/CMS_COND_34X_ECAL")           
          )
)

process.read = cms.EDAnalyzer("EcalDQMStatusReader",
  verbose = cms.untracked.bool(True),
)

process.p = cms.Path(process.read)

