
import FWCore.ParameterSet.Config as cms

from Geometry.CaloEventSetup.CaloGeometryDBReader_cfi import *


from CondCore.DBCommon.CondDBCommon_cfi import *

PoolDBESSource = cms.ESSource("PoolDBESSource",
                              CondDBCommon,
                              loadAll = cms.bool(True),
                              toGet = cms.VPSet(
    cms.PSet(record = cms.string('PEcalBarrelRcd'   ),tag = cms.string('TEST02')),
    cms.PSet(record = cms.string('PEcalEndcapRcd'   ),tag = cms.string('TEST03')),
    cms.PSet(record = cms.string('PEcalPreshowerRcd'),tag = cms.string('TEST04')),
    cms.PSet(record = cms.string('PHcalRcd'         ),tag = cms.string('TEST05')),
    cms.PSet(record = cms.string('PCaloTowerRcd'    ),tag = cms.string('TEST06')),
    cms.PSet(record = cms.string('PZdcRcd'          ),tag = cms.string('TEST07')),
    cms.PSet(record = cms.string('PCastorRcd'       ),tag = cms.string('TEST08'))
    ),
                              BlobStreamerName = cms.untracked.string('TBufferBlobStreamingService'),
                              timetype = cms.untracked.string('runnumber'),
                              connect = cms.string('sqlite_file:calofile.db')
                              )

