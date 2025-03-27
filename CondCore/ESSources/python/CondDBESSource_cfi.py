#This is the default configuration for the connection to the frontier servlets
#in order to fetch the condition payloads in CMSSW.
import socket
import FWCore.ParameterSet.Config as cms
from CondCore.CondDB.CondDB_cfi import *

CondDBConnection = CondDB.clone( connect = cms.string( 'frontier://FrontierProd/CMS_CONDITIONS' ) )
from CondCore.ESSources.default_CondDBESource_cfi import PoolDBESSource as _PoolDBESSource

GlobalTag = _PoolDBESSource(
    CondDBConnection,
    globaltag        = '',
    snapshotTime     = '',
    frontierKey      = '',
    toGet            = [],   # hook to override or add single payloads
    JsonDumpFileName = '',
    DumpStat         = False,
    ReconnectEachRun = False,
    RefreshAlways    = False,
    RefreshEachRun   = False,
    RefreshOpenIOVs  = False,
    pfnPostfix       = '',
    pfnPrefix        = '' ,
)
