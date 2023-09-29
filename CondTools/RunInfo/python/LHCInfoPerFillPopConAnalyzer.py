import socket
import FWCore.ParameterSet.Config as cms
import FWCore.ParameterSet.VarParsing as VarParsing
process = cms.Process("LHCInfoPerFillPopulator")
from CondCore.CondDB.CondDB_cfi import *
#process.load("CondCore.DBCommon.CondDBCommon_cfi")
#process.CondDBCommon.connect = 'sqlite_file:lhcinfoperls_pop_test.db'
#process.CondDBCommon.DBParameters.authenticationPath = '.'
#process.CondDBCommon.DBParameters.messageLevel=cms.untracked.int32(1)

sourceConnection = 'oracle://cms_omds_adg/CMS_RUNINFO_R'
if socket.getfqdn().find('.cms') != -1:
    sourceConnection = 'oracle://cms_omds_lb/CMS_RUNINFO_R'

options = VarParsing.VarParsing()
options.register( 'mode'
                , None # Required parameter
                , VarParsing.VarParsing.multiplicity.singleton
                , VarParsing.VarParsing.varType.string
                , "The mode the fills are going to be process and the data gathered. Accepted values: duringFill endFill"
                  )
options.register( 'destinationConnection'
                , 'sqlite_file:lhcinfo_pop_test.db' #default value
                , VarParsing.VarParsing.multiplicity.singleton
                , VarParsing.VarParsing.varType.string
                , "Connection string to the DB where payloads will be possibly written."
                  )
options.register( 'targetConnection'
                , '' #default value
                , VarParsing.VarParsing.multiplicity.singleton
                , VarParsing.VarParsing.varType.string
                , """Connection string to the target DB:
                     if not empty (default), this provides the latest IOV and payloads to compare;
                     it is the DB where payloads should be finally uploaded."""
                  )
options.register( 'tag'
                , 'LHCInfoPerFill_PopCon_test'
                , VarParsing.VarParsing.multiplicity.singleton
                , VarParsing.VarParsing.varType.string
                , "Tag written in destinationConnection and finally appended in targetConnection."
                  )
options.register( 'messageLevel'
                , 0 #default value
                , VarParsing.VarParsing.multiplicity.singleton
                , VarParsing.VarParsing.varType.int
                , "Message level; default to 0"
                  )
options.register( 'startTime'
                , '2021-09-10 03:10:18.000'
                , VarParsing.VarParsing.multiplicity.singleton
                , VarParsing.VarParsing.varType.string
                , """Date and time of the start of processing:
                     processes only fills starting at startTime or later"""
                  )
options.register( 'endTime'
                , ''
                , VarParsing.VarParsing.multiplicity.singleton
                , VarParsing.VarParsing.varType.string
                , """Date and time of the start of processing:
                     processes only fills starting before endTime;
                     default to empty string which sets no restriction"""
                  )
options.parseArguments()
if options.mode is None:
  raise ValueError("mode argument not provided. Supported modes are: duringFill endFill")
if options.mode not in ("duringFill", "endFill"):
  raise ValueError("Wrong mode argument. Supported modes are: duringFill endFill")

CondDBConnection = CondDB.clone( connect = cms.string( options.destinationConnection ) )
CondDBConnection.DBParameters.messageLevel = cms.untracked.int32( options.messageLevel )

process.MessageLogger = cms.Service("MessageLogger",
                                    cout = cms.untracked.PSet(threshold = cms.untracked.string('INFO')),
                                    destinations = cms.untracked.vstring('cout')
                                    )

process.source = cms.Source("EmptyIOVSource",
                            lastValue = cms.uint64(1),
                            timetype = cms.string('runnumber'),
                            firstValue = cms.uint64(1),
                            interval = cms.uint64(1)
                            )

process.PoolDBOutputService = cms.Service("PoolDBOutputService",
                                          CondDBConnection,
                                          timetype = cms.untracked.string('timestamp'),
                                          toPut = cms.VPSet(cms.PSet(record = cms.string('LHCInfoPerFillRcd'),
                                                                     tag = cms.string( options.tag )
                                                                     )
                                                            )
                                          )

process.Test1 = cms.EDAnalyzer("LHCInfoPerFillPopConAnalyzer",
                               SinceAppendMode = cms.bool(True),
                               record = cms.string('LHCInfoPerFillRcd'),
                               name = cms.untracked.string('LHCInfo'),
                               Source = cms.PSet(fill = cms.untracked.uint32(6417),
                                   startTime = cms.untracked.string(options.startTime),
                                   endTime = cms.untracked.string(options.endTime),
                                   samplingInterval = cms.untracked.uint32( 600 ),
                                   endFill = cms.untracked.bool(True if options.mode == "endFill" else False),
                                   name = cms.untracked.string("LHCInfoPerFillPopConSourceHandler"),
                                   connectionString = cms.untracked.string("oracle://cms_orcon_adg/CMS_RUNTIME_LOGGER"),
                                   ecalConnectionString = cms.untracked.string("oracle://cms_orcon_adg/CMS_DCS_ENV_PVSS_COND"),
                                   DIPSchema = cms.untracked.string("CMS_BEAM_COND"),
                                   omsBaseUrl = cms.untracked.string("http://vocms0184.cern.ch/agg/api/v1"),
                                   authenticationPath = cms.untracked.string(""),
                                   debug=cms.untracked.bool(False)
                                                 ),
                               loggingOn = cms.untracked.bool(True),
                               IsDestDbCheckedInQueryLog = cms.untracked.bool(False)
                               )

process.p = cms.Path(process.Test1)
