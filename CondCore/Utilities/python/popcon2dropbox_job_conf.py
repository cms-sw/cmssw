import FWCore.ParameterSet.Config as cms
import FWCore.ParameterSet.VarParsing as VarParsing
import popcon2dropbox

options = VarParsing.VarParsing()
options.register('popconConfigFileName',
                'popcon2dropbox.json',
                VarParsing.VarParsing.multiplicity.singleton,
                VarParsing.VarParsing.varType.string,
                "PopCon config file name")

options.parseArguments()

md = popcon2dropbox.CondMetaData( options.popconConfigFileName )

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

destinationDatabase = md.destinationDatabase()


process = cms.Process("PopCon")
process.load("CondCore.CondDB.CondDB_cfi")
process.CondDB.DBParameters.messageLevel = cms.untracked.int32( 3 )
process.CondDB.connect = 'sqlite:%s' %popcon2dropbox.dbFileForDropBox

process.PoolDBOutputService = cms.Service("PoolDBOutputService",
    process.CondDB,
    toPut = cms.VPSet( psetForOutRec )
)

process.source = cms.Source("EmptyIOVSource",
    timetype   = cms.string('runnumber'),
    firstValue = cms.uint64(1),
    lastValue  = cms.uint64(1),
    interval   = cms.uint64(1)
)


