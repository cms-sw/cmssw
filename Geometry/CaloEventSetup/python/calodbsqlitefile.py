
import FWCore.ParameterSet.Config as cms
 
idealGeometryRecord = cms.ESSource("EmptyESSource",
                                   recordName = cms.string('IdealGeometryRecord'),
                                   iovIsRunNotTime = cms.bool(True),
                                   firstValid = cms.vuint32(1)
                                   )

from CondCore.DBCommon.CondDBCommon_cfi import *

PoolDBESSource = cms.ESSource("PoolDBESSource",
                              CondDBCommon,
                              loadAll = cms.bool(True),
                              toGet = cms.VPSet(
    cms.PSet( record = cms.string('PEcalBarrelRcd'   ),
              tag = cms.string('EBRECO_Geometry_Test01')),
    cms.PSet( record = cms.string('PEcalEndcapRcd'   ),
              tag = cms.string('EERECO_Geometry_Test01')),
    cms.PSet( record = cms.string('PEcalPreshowerRcd'),
              tag = cms.string('EPRECO_Geometry_Test01')),
    cms.PSet( record = cms.string('PHcalRcd'         ),
              tag = cms.string('HCALRECO_Geometry_Test01')),
    cms.PSet( record = cms.string('PCaloTowerRcd'    ),
              tag = cms.string('CTRECO_Geometry_Test01')),
    cms.PSet( record = cms.string('PZdcRcd'          ),
              tag = cms.string('ZDCRECO_Geometry_Test01')),
    cms.PSet( record = cms.string('PCastorRcd'       ),
              tag = cms.string('CASTORRECO_Geometry_Test01'))
    ),
                              BlobStreamerName = cms.untracked.string('TBufferBlobStreamingService'),
                              timetype = cms.untracked.string('runnumber')
                              )

PoolDBESSource.connect = cms.string('sqlite_file:calofile.db')
