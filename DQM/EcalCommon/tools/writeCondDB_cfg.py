import FWCore.ParameterSet.Config as cms
from FWCore.ParameterSet.VarParsing import VarParsing
from DQM.EcalCommon.CalibCommonParams_cfi import ecalCalibCommonParams
import os

options = VarParsing('analysis')
options.register('source', '', VarParsing.multiplicity.singleton, VarParsing.varType.string, '')
options.register('runtype', '', VarParsing.multiplicity.singleton, VarParsing.varType.string, '')
options.register('dbparams', '', VarParsing.multiplicity.singleton, VarParsing.varType.string, '')
options.register('laserwl', '', VarParsing.multiplicity.list, VarParsing.varType.int, '')
options.register('ledwl', '', VarParsing.multiplicity.list, VarParsing.varType.int, '')
options.register('gains', '', VarParsing.multiplicity.list, VarParsing.varType.int, '')
options.register('pngains', '', VarParsing.multiplicity.list, VarParsing.varType.int, '')

options.parseArguments()

source = options.source
runtype = options.runtype
laserwl = []
if len(options.laserwl) > 0:
    laserwl = options.laserwl
else:
    laserwl = ecalCalibCommonParams['laserWavelengths']
ledwl = []
if len(options.ledwl) > 0:
    ledwl = options.ledwl
else:
    ledwl = ecalCalibCommonParams['ledWavelengths']
gains = []
if len(options.gains) > 0:
    gains = options.gains
else:
    gains = ecalCalibCommonParams['MGPAGains']
pngains = []
if len(options.pngains) > 0:
    pngains = options.pngains
else:
    pngains = ecalCalibCommonParams['MGPAGainsPN']

os.environ["TNS_ADMIN"] = "/etc"

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

process =  cms.Process("DQMDB")

process.source = cms.Source("EmptySource")

process.load("DQM.EcalCommon.EcalCondDBWriter_cfi")
process.ecalCondDBWriter.DBName = dbName
process.ecalCondDBWriter.userName = dbUserName
process.ecalCondDBWriter.password = dbPassword
process.ecalCondDBWriter.hostName = dbHostName
process.ecalCondDBWriter.hostPort = dbHostPort
process.ecalCondDBWriter.location = 'P5_Co'
process.ecalCondDBWriter.runType = runtype
process.ecalCondDBWriter.runGeneralTag = "LOCAL"
process.ecalCondDBWriter.monRunGeneralTag = 'CMSSW-online-private'
process.ecalCondDBWriter.inputRootFiles = cms.untracked.vstring(
    source
)
process.ecalCondDBWriter.workerParams.laserWavelengths = laserwl
process.ecalCondDBWriter.workerParams.ledWavelengths = ledwl
process.ecalCondDBWriter.workerParams.MGPAGains = gains
process.ecalCondDBWriter.workerParams.MGPAGainsPN = pngains
process.ecalCondDBWriter.verbosity = 2

process.load("DQM.EcalCommon.EcalDQMBinningService_cfi")

process.load("DQMServices.Core.DQM_cfg")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)

process.load("Geometry.CaloEventSetup.CaloGeometry_cfi")

process.load("Geometry.CaloEventSetup.CaloTopology_cfi")

process.load("Geometry.CaloEventSetup.EcalTrigTowerConstituents_cfi")

process.load("Geometry.CMSCommonData.cmsIdealGeometryXML_cfi")

process.load("Geometry.EcalMapping.EcalMapping_cfi")

process.load("Geometry.EcalMapping.EcalMappingRecord_cfi")

process.load("DQM.Integration.test.FrontierCondition_GT_cfi")

process.p = cms.Path(process.ecalCondDBWriter)
