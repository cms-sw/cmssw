import FWCore.ParameterSet.Config as cms

#toPut = cms.VPSet( [ cms.PSet( record = k,
#                               tag = v.get('destinationTag'),
#                               timetype = v.get('timetype')
#                               ) for k,v in md.outputRecords().items()] )

import popcon2dropbox
md = popcon2dropbox.CondMetaData()
psetForRec = []
for k,v in md.records().items():
    psetForRec.append( cms.PSet( record = cms.string(str(k)),
                                 tag = cms.string(str(v.get('destinationTag'))),
                                 ) 
                       )
print '####  1'
    
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

process.essource = cms.ESSource("PoolDBESSource",
                                #connect = cms.string('oracle://cms_orcon_prod/CMS_COND_31X_DT'),
                                connect = cms.string( str(md.destinationDatabase()) ),
                                DBParameters = cms.PSet( authenticationPath = cms.untracked.string( str(md.authPath()) ),
                                                         authenticationSystem = cms.untracked.int32( int(md.authSys()) )
                                                         ),
                                DumpStat=cms.untracked.bool(True),
                                toGet = cms.VPSet( psetForRec )
)
