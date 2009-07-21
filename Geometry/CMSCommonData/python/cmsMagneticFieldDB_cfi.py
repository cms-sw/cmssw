# The following comments couldn't be translated into the new config version:

# label for IOV to use.
# connection string only needed for name-to-token.
# should be optional, then check if needed by dbConn
# "ditto"
# empty string defaults to Top Node of DDD
import FWCore.ParameterSet.Config as cms

magfield = cms.ESSource("DBIdealGeometryESSource",
    dbUser = cms.string('mcase'),
    dbPass = cms.string('neneng'),
    dbMetaName = cms.string('MagField01'),
    dbConn = cms.string('sqlite_file:testMag.db'),
    rootNodeName = cms.string('MagneticFieldVolumes:MAGF')
)


