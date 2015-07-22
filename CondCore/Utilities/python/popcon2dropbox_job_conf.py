import FWCore.ParameterSet.Config as cms

import popcon2dropbox
md = popcon2dropbox.CondMetaData()
psetForRec = []
for k,v in md.records().items():
    psetForRec.append( cms.PSet( record = cms.string(str(k)),
                                 tag = cms.string(str(v.get('destinationTag'))),
                                 ) 
                       )
    
psetForOutRec = []
for k,v in md.records().items():
        outRec = v.get('outputRecord')
        if outRec == None:
            outRec = k
        sqliteTag = v.get('sqliteTag')
        if sqliteTag == None:
            sqliteTag = v.get('destinationTag')
        psetForOutRec.append( cms.PSet( record = cms.string(str( outRec )),
                                        tag = cms.string(str( sqliteTag )),
                                        timetype = cms.untracked.string(str(v.get('timetype')))
                                        )
                              )
print psetForOutRec

process = cms.Process("TEST")
process.load("CondCore.CondDB.CondDB_cfi")
process.CondDB.DBParameters.messageLevel = cms.untracked.int32( 3 )
process.CondDB.connect = 'sqlite:%s' %popcon2dropbox.dbFileForDropBox

process.PoolDBOutputService = cms.Service("PoolDBOutputService",
    process.CondDB,
    logconnect = cms.untracked.string('sqlite:%s' %popcon2dropbox.dbLogFile),
    toPut = cms.VPSet( psetForOutRec )
)

process.source = cms.Source("EmptyIOVSource",
    timetype   = cms.string('runnumber'),
    firstValue = cms.uint64(1),
    lastValue  = cms.uint64(1),
    interval   = cms.uint64(1)
)

print process.CondDB.connect

