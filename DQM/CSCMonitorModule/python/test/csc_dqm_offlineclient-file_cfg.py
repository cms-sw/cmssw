import FWCore.ParameterSet.Config as cms

process = cms.Process("CSCDQM")

#-------------------------------------------------
# DQM Environment Configuration
#-------------------------------------------------

process.load("DQMServices.Components.MEtoEDMConverter_cff")
process.load("DQMServices.Components.DQMEnvironment_cfi")

#-------------------------------------------------
# DQM Module Configuration
#-------------------------------------------------

process.load("DQM.CSCMonitorModule.csc_dqm_offlineclient_cosmics_cff")

#----------------------------
# Event Source
#-----------------------------

process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(1))

process.options = cms.untracked.PSet(
 fileMode = cms.untracked.string('FULLMERGE')
)

#process.source = cms.Source("PoolSource",
#  fileNames  = cms.untracked.vstring('file:Playback_V0001_CSC_R000110388.root')
#)

process.source = cms.Source("EmptySource")

process.reader = cms.EDFilter("DQMReadFileExample",
  #RootFileName = cms.untracked.string('Playback_V0001_CSC_R000110388.root')
  #RootFileName = cms.untracked.string('Playback_V0001_CSC_R000112389.root')
  RootFileName = cms.untracked.string('/tmp/valdo/Playback_V0001_CSC_R000143227.root')
)


#----------------------------
# DQM Environment
#-----------------------------

#process.dqmSaver.convention = 'Offline'
process.dqmSaver.workflow = '/testing/damn/reader'
process.dqmSaver.dirName = '/tmp/valdo'
process.DQMStore.collateHistograms = False
process.DQMStore.referenceFileName = ''
#process.DQMStore.referenceFileName = '/afs/cern.ch/user/v/valdo/data/csc_reference.root'

#-------------------------------------------------
# GEOMETRY
#-------------------------------------------------

process.load("Configuration.StandardSequences.GeometryRecoDB_cff")

#-------------------------------------------------
# Global Tag
#-------------------------------------------------

#process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
#process.GlobalTag.globaltag = 'GR09_31X_V1P::All' 
#process.es_prefer_GlobalTag = cms.ESPrefer('PoolDBESSource','GlobalTag')

#--------------------------
# Message Logger
#--------------------------

MessageLogger = cms.Service("MessageLogger",

# suppressInfo = cms.untracked.vstring('source'),
# suppressInfo = cms.untracked.vstring('*'),

  cout = cms.untracked.PSet(
    threshold = cms.untracked.string('DEBUG'),
#    WARNING = cms.untracked.PSet(
#      limit = cms.untracked.int32(0)
#    ),
#    noLineBreaks = cms.untracked.bool(False)
  ),

  detailedInfo = cms.untracked.PSet(
    threshold = cms.untracked.string('DEBUG')
  ),

#  critical = cms.untracked.PSet(
#    threshold = cms.untracked.string('ERROR')
#  ),

  debugModules = cms.untracked.vstring('*'),

  destinations = cms.untracked.vstring(
    'detailedInfo', 
    'critical', 
    'cout'
  )

)

#--------------------------
# Sequences
#--------------------------

process.p = cms.Path(process.reader * process.cscOfflineCosmicsClients + process.dqmSaver)
#process.p = cms.Path(process.EDMtoMEConverter*process.dqmCSCOfflineClient*process.dqmSaver)


