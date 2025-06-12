
import FWCore.ParameterSet.Config as cms

from CondCore.DBCommon.CondDBCommon_cfi import *

PoolDBESSource = cms.ESSource( "PoolDBESSource",
                               CondDBCommon,
                               toGet = cms.VPSet(
    cms.PSet( record = cms.string('GeometryFileRcd'   ),
              tag = cms.string('XMLFILE_Geometry_Test03'))    ),
                               )

PoolDBESSource.connect = cms.string('sqlite_file:allxml.db')
