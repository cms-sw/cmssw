import socket
import FWCore.ParameterSet.Config as cms
import FWCore.ParameterSet.VarParsing as VarParsing
process = cms.Process("LHCInfoPerLSPopulator")
from CondCore.CondDB.CondDB_cfi import *

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
                , 'LHCInfoPerLS_PopCon_test'
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

options.register( 'sourceConnection'
                , "oracle://cms_orcon_adg/CMS_RUNTIME_LOGGER"
                , VarParsing.VarParsing.multiplicity.singleton
                , VarParsing.VarParsing.varType.string
                , """beam data source connection string (aka PPS db)
                     It's the source of crossing angle and beta * data"""
                  )
options.register( 'oms'
                , "http://vocms0184.cern.ch/agg/api/v1"
                , VarParsing.VarParsing.multiplicity.singleton
                , VarParsing.VarParsing.varType.string
                , """OMS base URL"""
                  )

#duringFill mode specific:
options.register( 'lastLumiFile'
                , ''
                , VarParsing.VarParsing.multiplicity.singleton
                , VarParsing.VarParsing.varType.string
                , """duringFill only: path to file with lumiid to override the last lumisection processed by HLT.
                     Used for testing. Leave empty for production behaviour (getting this info from OMS)"""
                  )
options.register( 'frontierKey'
                , ''
                , VarParsing.VarParsing.multiplicity.singleton
                , VarParsing.VarParsing.varType.string
                , """duringFill only: run-unique key for writing with OnlinePopCon
                     (used for confirming proper upload)"""
                  )
options.register('offsetLS'
                , 2
                , VarParsing.VarParsing.multiplicity.singleton
                , VarParsing.VarParsing.varType.int
                , """duringFill only: offset between lastLumi (last LS processed by HLT or overriden by lastLumiFile)
                     and the IOV of the payload to be uploaded"""
                  )
options.register( 'debugLogic'
                , False
                , VarParsing.VarParsing.multiplicity.singleton
                , VarParsing.VarParsing.varType.bool
                , """duringFill only: Enables debug logic, meant to be used only for tests"""
                  )
options.register( 'defaultXangleX'
                , 160.0 # urad
                , VarParsing.VarParsing.multiplicity.singleton
                , VarParsing.VarParsing.varType.float
                , """duringFill only: crossingAngleX value (in urad) for the default payload.
                     The default payload is inserted after the last processed fill has ended
                     and there's no ongoing stable beam yet. """
                  )
options.register( 'defaultXangleY'
                , 0.0 # urad
                , VarParsing.VarParsing.multiplicity.singleton
                , VarParsing.VarParsing.varType.float
                , """duringFill only: crossingAngleY value (in urad) for the default payload.
                     The default payload is inserted after the last processed fill has ended
                     and there's no ongoing stable beam yet. """
                  )
options.register( 'defaultBetaX'
                , 0.3 # meters
                , VarParsing.VarParsing.multiplicity.singleton
                , VarParsing.VarParsing.varType.float
                , """duringFill only: betaStarX value (in meters) for the default payload.
                     The default payload is inserted after the last processed fill has ended
                     and there's no ongoing stable beam yet. """
                  )
options.register( 'defaultBetaY'
                , 0.3 # meters
                , VarParsing.VarParsing.multiplicity.singleton
                , VarParsing.VarParsing.varType.float
                , """duringFill only: betaStarY value (in meters) for the default payload.
                     The default payload is inserted after the last processed fill has ended
                     and there's no ongoing stable beam yet. """
                  )


# it's unlikely to ever use values different from the defaults, added as a parameter just in case
options.register('minBetaStar',  0.1 
                , VarParsing.VarParsing.multiplicity.singleton, VarParsing.VarParsing.varType.float
                , """duringFill only: [meters] min value of the range of valid values.
                     If the value is outside of this range the payload is not uploaded""")
options.register('maxBetaStar',  100.
                , VarParsing.VarParsing.multiplicity.singleton, VarParsing.VarParsing.varType.float
                , """duringFill only: [meters] min value of the range of valid values.
                     If the value is outside of this range the payload is not uploaded""")
options.register('minCrossingAngle',  10.
                , VarParsing.VarParsing.multiplicity.singleton, VarParsing.VarParsing.varType.float
                , """duringFill only: [urad] min value of the range of valid values.
                     If the value is outside of this range the payload is not uploaded""")
options.register('maxCrossingAngle',  500.
                , VarParsing.VarParsing.multiplicity.singleton, VarParsing.VarParsing.varType.float
                , """duringFill only: [urad] min value of the range of valid values.
                     If the value is outside of this range the payload is not uploaded""")

# as the previous options, so far there was no need to use option, added just in case
options.register( 'authenticationPath'
                , ""
                , VarParsing.VarParsing.multiplicity.singleton
                , VarParsing.VarParsing.varType.string
                , """for now this option was always left empty"""
                  )

options.parseArguments()
if options.mode is None:
  raise ValueError("mode argument not provided. Supported modes are: duringFill endFill")
if options.mode not in ("duringFill", "endFill"):
  raise ValueError("Wrong mode argument. Supported modes are: duringFill endFill")

CondDBConnection = CondDB.clone( connect = cms.string( options.destinationConnection ) )
CondDBConnection.DBParameters.messageLevel = cms.untracked.int32( options.messageLevel )
CondDBConnection.DBParameters.authenticationPath = cms.untracked.string(options.authenticationPath)

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

# Write different time-types tags depending on the O2O mode
if options.mode == 'endFill':
  timetype = 'timestamp'
else:
  timetype = 'lumiid'

if options.mode == "endFill":
  process.PoolDBOutputService = cms.Service("PoolDBOutputService",
                                            CondDBConnection,
                                            timetype = cms.untracked.string(timetype),
                                            toPut = cms.VPSet(cms.PSet(record = cms.string('LHCInfoPerLSRcd'),
                                                                      tag = cms.string( options.tag )
                                                                      )
                                                              )
                                            )
else:
  process.OnlineDBOutputService = cms.Service("OnlineDBOutputService",
    CondDBConnection,
    preLoadConnectionString = cms.untracked.string('frontier://FrontierProd/CMS_CONDITIONS' 
                                                   if not options.destinationConnection.startswith('sqlite') 
                                                   else options.destinationConnection ),
    lastLumiFile = cms.untracked.string(options.lastLumiFile),
    omsServiceUrl = cms.untracked.string('http://cmsoms-eventing.cms:9949/urn:xdaq-application:lid=100/getRunAndLumiSection'
                                         if not options.lastLumiFile else "" ),
    # runNumber = cms.untracked.uint64(384468), #not used in production, the last LS processed is set as the 1st LS of this
                                                #run if the omsServiceUrl is empty and file specified in lastLumiFile is empty
    latency = cms.untracked.uint32(options.offsetLS),
    timetype = cms.untracked.string(timetype),
    toPut = cms.VPSet(cms.PSet(
        record = cms.string('LHCInfoPerLSRcd'),
        tag = cms.string( options.tag ),
        onlyAppendUpdatePolicy = cms.untracked.bool(True)
    )),
    frontierKey = cms.untracked.string(options.frontierKey)
)


process.Test1 = cms.EDAnalyzer(("LHCInfoPerLSPopConAnalyzer" if options.mode == "endFill" 
                               else "LHCInfoPerLSOnlinePopConAnalyzer"),
                               SinceAppendMode = cms.bool(True),
                               record = cms.string('LHCInfoPerLSRcd'),
                               name = cms.untracked.string('LHCInfo'),
                               Source = cms.PSet(
                                   startTime = cms.untracked.string(options.startTime),
                                   endTime = cms.untracked.string(options.endTime),
                                   endFill = cms.untracked.bool(options.mode == "endFill"),
                                   name = cms.untracked.string("LHCInfoPerLSPopConSourceHandler"),
                                   connectionString = cms.untracked.string(options.sourceConnection),
                                   omsBaseUrl = cms.untracked.string(options.oms),
                                   authenticationPath = cms.untracked.string(options.authenticationPath),
                                   debug=cms.untracked.bool(False), # Additional logs
                                   debugLogic=cms.untracked.bool(options.debugLogic),
                                   defaultCrossingAngleX = cms.untracked.double(options.defaultXangleX),
                                   defaultCrossingAngleY = cms.untracked.double(options.defaultXangleY),
                                   defaultBetaStarX = cms.untracked.double(options.defaultBetaX),
                                   defaultBetaStarY = cms.untracked.double(options.defaultBetaY),
                                   minBetaStar = cms.untracked.double(options.minBetaStar),
                                   maxBetaStar = cms.untracked.double(options.maxBetaStar),
                                   minCrossingAngle = cms.untracked.double(options.minCrossingAngle),
                                   maxCrossingAngle = cms.untracked.double(options.maxCrossingAngle),
                               ),
                               loggingOn = cms.untracked.bool(True),
                               IsDestDbCheckedInQueryLog = cms.untracked.bool(False)
                               )

process.p = cms.Path(process.Test1)