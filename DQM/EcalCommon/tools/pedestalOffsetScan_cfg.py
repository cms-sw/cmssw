import FWCore.ParameterSet.Config as cms
from FWCore.ParameterSet.VarParsing import VarParsing

options = VarParsing('analysis')
options.register('num', '', VarParsing.multiplicity.singleton, VarParsing.varType.int, '')
options.register('runnum', '', VarParsing.multiplicity.singleton, VarParsing.varType.string, '')
options.register('dbparams', '', VarParsing.multiplicity.singleton, VarParsing.varType.string, '')

options.parseArguments()

num = options.num
runnum = options.runnum

import os
os.environ['TNS_ADMIN']='/etc'

dbName = ''
dbHostName = ''
dbHostPort = 1521
dbUserName = ''
dbPassword = ''

try:
  file = open(options.dbparams, 'r')
  for line in file:
    if line.find('dbName') >= 0:
      dbName = line.split()[2]
    if line.find('dbHostName') >= 0:
      dbHostName = line.split()[2]
    if line.find('dbHostPort') >= 0:
      dbHostPort = int(line.split()[2])
    if line.find('dbUserName') >= 0:
      dbUserName = line.split()[2]
    if line.find('dbPassword') >= 0:
      dbPassword = line.split()[2]
  file.close()
except IOError:
  pass

process = cms.Process("ECALDQM")

process.load("EventFilter.EcalRawToDigi.EcalUnpackerMapping_cfi")

process.load("EventFilter.EcalRawToDigi.EcalUnpackerData_cfi")

process.load("Geometry.EcalMapping.EcalMapping_cfi")

process.load("Geometry.EcalMapping.EcalMappingRecord_cfi")

process.load("CalibCalorimetry.EcalTrivialCondModules.EcalTrivialCondRetriever_cfi")

process.ecalPedOffset = cms.EDAnalyzer("EcalPedOffset",
    EBdigiCollection = cms.InputTag("ecalEBunpacker","ebDigis"),
    EEdigiCollection = cms.InputTag("ecalEBunpacker","eeDigis"),
    headerCollection = cms.InputTag("ecalEBunpacker"),

    DACmin = cms.untracked.int32(40),
    DACmax = cms.untracked.int32(90),
    RMSmax = cms.untracked.double(20.0),

    bestPed = cms.untracked.int32(200),
    minSlopeAllowed = cms.untracked.double(-18.0),
    maxSlopeAllowed = cms.untracked.double(-29.0),
    maxChi2OverNDF = cms.untracked.double(5.25),

    dbName = cms.untracked.string(dbName),
    dbHostName = cms.untracked.string(dbHostName),
    dbHostPort = cms.untracked.int32(dbHostPort),
    dbUserName = cms.untracked.string(dbUserName),
    dbPassword = cms.untracked.string(dbPassword),

    createMonIOV = cms.untracked.bool(False),
    location = cms.untracked.string('P5_Co'),
    run = cms.int32(num),
    xmlFile = cms.string('pedestal-offset-' + runnum),
    plotting = cms.string('pedestal-offset-' + runnum)
)

process.printAscii = cms.OutputModule("AsciiOutputModule",
    prescale = cms.untracked.uint32(10)
)

process.source = cms.Source("NewEventStreamFileReader",
    fileNames = cms.untracked.vstring(
        options.inputFiles
    )
)

process.MessageLogger = cms.Service("MessageLogger",
    cout = cms.untracked.PSet(
        threshold = cms.untracked.string('WARNING'),
        noLineBreaks = cms.untracked.bool(True),
        noTimeStamps = cms.untracked.bool(True),
        default = cms.untracked.PSet(
            limit = cms.untracked.int32(0)
        ),
        EcalRawToDigi = cms.untracked.PSet(
            limit = cms.untracked.int32(1000)
        ),
        EcalRawToDigiTriggerType = cms.untracked.PSet(
            limit = cms.untracked.int32(1000)
        ),
        EcalRawToDigiTpg = cms.untracked.PSet(
            limit = cms.untracked.int32(1000)
        ),
        EcalRawToDigiNumTowerBlocks = cms.untracked.PSet(
            limit = cms.untracked.int32(1000)
        ),
        EcalRawToDigiTowerId = cms.untracked.PSet(
            limit = cms.untracked.int32(1000)
        ),
        EcalRawToDigiTowerSize = cms.untracked.PSet(
            limit = cms.untracked.int32(1000)
        ),
        EcalRawToDigiChId = cms.untracked.PSet(
            limit = cms.untracked.int32(1000)
        ),
        EcalRawToDigiGainZero = cms.untracked.PSet(
            limit = cms.untracked.int32(1000)
        ),
        EcalRawToDigiGainSwitch = cms.untracked.PSet(
            limit = cms.untracked.int32(1000)
        ),
        EcalRawToDigiDccBlockSize = cms.untracked.PSet(
            limit = cms.untracked.int32(1000)
        ),
        EcalRawToDigiMemBlock = cms.untracked.PSet(
            limit = cms.untracked.int32(1000)
        ),
        EcalRawToDigiMemTowerId = cms.untracked.PSet(
            limit = cms.untracked.int32(1000)
        ),
        EcalRawToDigiMemChId = cms.untracked.PSet(
            limit = cms.untracked.int32(1000)
        ),
        EcalRawToDigiMemGain = cms.untracked.PSet(
            limit = cms.untracked.int32(1000)
        ),
        EcalRawToDigiTCC = cms.untracked.PSet(
            limit = cms.untracked.int32(1000)
        ),
        EcalRawToDigiSRP = cms.untracked.PSet(
            limit = cms.untracked.int32(1000)
        ),
        EcalDCCHeaderRuntypeDecoder = cms.untracked.PSet(
            limit = cms.untracked.int32(1000)
        ),
        EcalPedOffset = cms.untracked.PSet(
            limit = cms.untracked.int32(1000)
        )
    ),
    categories = cms.untracked.vstring('EcalRawToDigi',
                                       'EcalRawToDigiTriggerType',
                                       'EcalRawToDigiTpg',
                                       'EcalRawToDigiNumTowerBlocks',
                                       'EcalRawToDigiTowerId',
                                       'EcalRawToDigiTowerSize',
                                       'EcalRawToDigiChId',
                                       'EcalRawToDigiGainZero',
                                       'EcalRawToDigiGainSwitch',
                                       'EcalRawToDigiDccBlockSize',
                                       'EcalRawToDigiMemBlock',
                                       'EcalRawToDigiMemTowerId',
                                       'EcalRawToDigiMemChId',
                                       'EcalRawToDigiMemGain',
                                       'EcalRawToDigiTCC',
                                       'EcalRawToDigiSRP',
                                       'EcalDCCHeaderRuntypeDecoder',
                                       'EcalPedOffset'),
    destinations = cms.untracked.vstring('cout')
)

process.p = cms.Path(process.ecalEBunpacker*process.ecalPedOffset)
process.q = cms.EndPath(process.printAscii)

process.ecalEBunpacker.silentMode = True
#process.ecalEBunpacker.InputLabel = cms.InputTag('rawDataCollector')
