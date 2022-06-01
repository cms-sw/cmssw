import FWCore.ParameterSet.Config as cms
import os

process = cms.Process("StripsO2O")

process.MessageLogger = cms.Service( "MessageLogger",
                                     debugModules = cms.untracked.vstring( "*" ),
                                     cout = cms.untracked.PSet( threshold = cms.untracked.string( "DEBUG" ) ),
                                     destinations = cms.untracked.vstring( "cout" )
                                     )

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32( 1 ) )

process.source = cms.Source( "EmptySource",
                             numberEventsInRun = cms.untracked.uint32(1),
                             firstRun = cms.untracked.uint32(1)
                             )


process.load("OnlineDB.SiStripConfigDb.SiStripConfigDb_cfi")
process.SiStripConfigDb.UsingDb = True
process.SiStripConfigDb.TNS_ADMIN = os.environ["TNS_ADMIN"]
process.SiStripConfigDb.Partitions = cms.untracked.PSet(
_CFGLINES_
)

if 'CONFDB' not in os.environ:
    import CondCore.Utilities.credentials as auth
    user, _, passwd = auth.get_credentials('cmsonr_lb/cms_trk_r')
    process.SiStripConfigDb.ConfDb = '{user}/{passwd}@{path}'.format(user=user, passwd=passwd, path='cmsonr_lb')

process.load("OnlineDB.SiStripO2O.SiStripO2OCalibrationFactors_cfi")
process.SiStripCondObjBuilderFromDb = cms.Service( "SiStripCondObjBuilderFromDb",
                                                   process.SiStripO2OCalibrationFactors
                                                   )
process.SiStripCondObjBuilderFromDb.UseFED = True
process.SiStripCondObjBuilderFromDb.UseFEC = True
process.SiStripCondObjBuilderFromDb.UseAnalysis = _USEANALYSIS_
process.SiStripCondObjBuilderFromDb.SiStripDetInfoFile = cms.FileInPath("CalibTracker/SiStripCommon/data/SiStripDetInfo.dat")
process.SiStripCondObjBuilderFromDb.SkippedDevices = cms.untracked.VPSet(
_SKIPPED_
)
process.SiStripCondObjBuilderFromDb.WhitelistedDevices = cms.untracked.VPSet(
_WHITELISTED_
)

process.load("CondCore.CondDB.CondDB_cfi")
process.siStripO2O = cms.EDAnalyzer( "_ANALYZER_",
                                     process.CondDB,
                                     configMapDatabase=cms.string("_HASHMAPDB_"),
                                     conditionDatabase=cms.string("_CONDDB_"),
                                     condDbFile=cms.string("_DBFILE_"),
                                     cfgMapDbFile=cms.string("_MAPDBFILE_"),
                                     targetTag=cms.string("_TARGETTAG_"),
                                     since=cms.uint32(_RUNNUMBER_)
                                     )

process.p = cms.Path(process.siStripO2O)
