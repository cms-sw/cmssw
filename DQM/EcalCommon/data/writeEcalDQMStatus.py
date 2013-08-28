import FWCore.ParameterSet.Config as cms

process = cms.Process("READ")

process.load("Geometry.CMSCommonData.cmsIdealGeometryXML_cfi")

process.load("Geometry.EcalMapping.EcalMapping_cfi")
process.load("Geometry.EcalMapping.EcalMappingRecord_cfi")

process.load("Geometry.CaloEventSetup.EcalTrigTowerConstituents_cfi")

process.load("CalibCalorimetry.EcalTrivialCondModules.EcalTrivialCondRetriever_cfi")
process.EcalTrivialConditionRetriever.producedEcalDQMChannelStatus = False
process.EcalTrivialConditionRetriever.producedEcalDQMTowerStatus = False

process.load("CondCore.DBCommon.CondDBCommon_cfi")
process.CondDBCommon.DBParameters.authenticationPath = '/nfshome0/popcondev/conddb'
process.CondDBCommon.connect = 'sqlite_file:mask-ECAL.db'

process.source = cms.Source("EmptyIOVSource",
                            firstValue = cms.uint64(1),
                            lastValue = cms.uint64(1),
                            timetype = cms.string('runnumber'),
                            interval = cms.uint64(1)
                            )

process.PoolDBOutputService = cms.Service("PoolDBOutputService",
  process.CondDBCommon,
  toPut = cms.VPSet(
    cms.PSet(
      record = cms.string('EcalDQMChannelStatusRcd'),
      tag = cms.string('EcalDQMChannelStatus_v1_hlt')
    ),
    cms.PSet(
      record = cms.string('EcalDQMTowerStatusRcd'),
      tag = cms.string('EcalDQMTowerStatus_v1_hlt')
    )
  )
)

process.write = cms.EDAnalyzer("EcalDQMStatusWriter",
  verbose = cms.untracked.bool(False),
  toPut = cms.VPSet(
#    cms.PSet(
#      conditionType = cms.untracked.string('EcalDQMChannelStatus'),
#      since = cms.untracked.uint32(1),
#      inputFile = cms.untracked.string('list.txt')
#    ),
    cms.PSet(
      conditionType = cms.untracked.string('EcalDQMChannelStatus'),
      since = cms.untracked.uint32(1),
      inputFile = cms.untracked.string('mask-ECAL.txt')
    ),
#    cms.PSet(
#      conditionType = cms.untracked.string('EcalDQMTowerStatus'),
#      since = cms.untracked.uint32(1),
#      inputFile = cms.untracked.string('list.txt')
#    ),
    cms.PSet(
      conditionType = cms.untracked.string('EcalDQMTowerStatus'),
      since = cms.untracked.uint32(1),
      inputFile = cms.untracked.string('mask-ECAL.txt')
    )
  )
)

process.p = cms.Path(process.write)

