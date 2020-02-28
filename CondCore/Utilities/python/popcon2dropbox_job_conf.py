from __future__ import absolute_import
import FWCore.ParameterSet.Config as cms
import FWCore.ParameterSet.VarParsing as VarParsing
from . import popcon2dropbox

options = VarParsing.VarParsing()
options.register('targetFile',
                'popcon.db',
                VarParsing.VarParsing.multiplicity.singleton,
                VarParsing.VarParsing.varType.string,
                "the target sqlite file name")
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

    sqliteConnect = 'sqlite:%s' %options.targetFile
    process = cms.Process("PopCon")
    process.load("CondCore.CondDB.CondDB_cfi")
    process.CondDB.DBParameters.messageLevel = cms.untracked.int32( 3 )

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
