
import FWCore.ParameterSet.Config as cms

from CondCore.DBCommon.CondDBCommon_cfi import *

PoolDBESSource = cms.ESSource( "PoolDBESSource",
                               CondDBCommon,
                               loadAll = cms.bool(True),
                               toGet = cms.VPSet(
    cms.PSet( record = cms.string('GeometryFileRcd'   ),
              tag = cms.string('XMLFILE_Geometry_Test03'))    ),
                               BlobStreamerName = cms.untracked.string('TBufferBlobStreamingService'),
                               timetype = cms.untracked.string('runnumber')
                               )

PoolDBESSource.connect = cms.string('sqlite_file:allxml.db')
