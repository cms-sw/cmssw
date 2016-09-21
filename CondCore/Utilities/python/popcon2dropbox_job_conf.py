import FWCore.ParameterSet.Config as cms
import FWCore.ParameterSet.VarParsing as VarParsing
import popcon2dropbox

options = VarParsing.VarParsing()
options.register('destinationDatabase',
                '',
                VarParsing.VarParsing.multiplicity.singleton,
                VarParsing.VarParsing.varType.string,
                "the destination database connection string")
options.register('destinationTag',
                '',
                VarParsing.VarParsing.multiplicity.singleton,
                VarParsing.VarParsing.varType.string,
                "the destination tag name")
options.parseArguments()

def setup_popcon( recordName, tagTimeType ):
    psetForOutRec = []
    psetForOutRec.append( cms.PSet( record = cms.string(str( recordName )),
                                    tag = cms.string(str( options.destinationTag )),
                                    timetype = cms.untracked.string(str(tagTimeType))
                                    )
                          )

    sqliteConnect = 'sqlite:%s' %popcon2dropbox.dbFileForDropBox
    process = cms.Process("PopCon")
    process.load("CondCore.CondDB.CondDB_cfi")
    process.CondDB.DBParameters.messageLevel = cms.untracked.int32( 3 )
    #process.CondDB.connect = 'sqlite:%s' %popcon2dropbox.dbFileForDropBox

    process.PoolDBOutputService = cms.Service("PoolDBOutputService",
                                              DBParameters = cms.PSet( messageLevel = cms.untracked.int32( 3 ),
                                                                       ),
                                              connect = cms.string( sqliteConnect ),
                                              toPut = cms.VPSet( psetForOutRec )
    )
    
    process.source = cms.Source("EmptyIOVSource",
                                timetype   = cms.string('runnumber'),
                                firstValue = cms.uint64(1),
                                lastValue  = cms.uint64(1),
                                interval   = cms.uint64(1)
    )
    return process

def psetForRecord( recordName ):
    psetForRec = []
    psetForRec.append( cms.PSet( record = cms.string(str(recordName)),
                                 tag = cms.string(str( options.destinationTag ))
                                 ) 
                       )
    return psetForRec
