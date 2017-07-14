###
### Read a geometry from a single xml file created from mfxmlwriter.py
### and write it into a db file.
###

import FWCore.ParameterSet.Config as cms

process = cms.Process("MagneticFieldWriter")
process.load("CondCore.DBCommon.CondDBCommon_cfi")


#GEOMETRY_VERSION = '90322'
#GEOMETRY_VERSION = '120812'
#GEOMETRY_VERSION = '130503'
GEOMETRY_VERSION = '160812'

process.source = cms.Source("EmptyIOVSource",
                            lastValue = cms.uint64(1),
                            timetype = cms.string('runnumber'),
                            firstValue = cms.uint64(1),
                            interval = cms.uint64(1)
                            )

# This reads the big XML file and the only way to fill the
# nonreco part of the database is to read this file.  It
# somewhat duplicates the information read from the little
# XML files, but there is no way to directly build the
# DDCompactView from this.
process.XMLGeometryWriter = cms.EDAnalyzer("XMLGeometryBuilder",
                                           XMLFileName = cms.untracked.string("./mfGeometry_"+GEOMETRY_VERSION+".xml"),
                                           ZIP = cms.untracked.bool(True),
                                           record = cms.untracked.string('MFGeometryFileRcd')
                                           )

process.CondDBCommon.BlobStreamerName = cms.untracked.string('TBufferBlobStreamingService')
process.CondDBCommon.timetype = cms.untracked.string('runnumber')
process.CondDBCommon.connect = cms.string('sqlite_file:mfGeometry_'+GEOMETRY_VERSION+'.db')
process.PoolDBOutputService = cms.Service("PoolDBOutputService",
                                          process.CondDBCommon,
                                          toPut = cms.VPSet(cms.PSet(record = cms.string('MFGeometryFileRcd'),tag = cms.string('MagneticFieldGeometry_'+str(GEOMETRY_VERSION))))
                                          )

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
    )

process.p1 = cms.Path(process.XMLGeometryWriter)


# Create the corresponding metadata file
f = open('mfGeometry_'+GEOMETRY_VERSION+'.txt','w')
f.write('{\n'+
        '    \"destinationDatabase\": \"oracle://cms_orcon_prod/CMS_CONDITIONS\",\n'+
        '    \"destinationTags\": {\n'+
        '       \"MFGeometry_'+GEOMETRY_VERSION+'\": {}\n'+
        '    },\n'+
        '    \"inputTag\": "MagneticFieldGeometry_'+GEOMETRY_VERSION+'\",\n'+
        '    \"since\": 1,\n'+
        '    \"userText\": "Mag field geometry, version '+GEOMETRY_VERSION+'\"\n'+
        '}\n'
        )
