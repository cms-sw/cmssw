import FWCore.ParameterSet.Config as cms

process = cms.Process('CALIB')
process.load('CalibTracker.Configuration.setupCalibrationTree_cff')
process.load('Configuration/StandardSequences/MagneticField_AutoFromDBCurrent_cff')
process.load('Configuration.Geometry.GeometryExtended2015Reco_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_condDBv2_cff')

#process.GlobalTag.globaltag = 'GR_P_V27::All' #2011
#process.GlobalTag.globaltag = 'GR_P_V32::All'  #2012AB #first run in 2012 is 190450
#process.GlobalTag.globaltag = 'GR_P_V40::All' #2012  
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'GLOBALTAG', '')

process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('Configuration.StandardSequences.Services_cff')
process.add_( cms.Service( "TFileService",
                           fileName = cms.string( 'OUTFILE' ),
                           closeFileFast = cms.untracked.bool(True)  ) )

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(-1) )
process.source = cms.Source (
    "PoolSource",
    fileNames = cms.untracked.vstring(FILES)
    )

process.options   = cms.untracked.PSet( wantSummary = cms.untracked.bool(True) )
process.MessageLogger.cerr.FwkReport.reportEvery = 10000

# Definition of the track collection module label
# To use the prompt reco dataset uncomment the following two lines
#process.CalibrationTracks.src = 'generalTracks'
#process.shallowTracks.Tracks  = 'generalTracks'
# To use the SiStripCalMinBias AlCaReco dataset uncomment the following two lines
process.CalibrationTracks.src = 'ALCARECOSiStripCalMinBias'
process.shallowTracks.Tracks  = 'ALCARECOSiStripCalMinBias'
# To use the cosmic reco dataset uncomment the following two lines
#process.CalibrationTracks.src = 'ctfWithMaterialTracksP5'
#process.shallowTracks.Tracks  = 'ctfWithMaterialTracksP5'

# BSCNoBeamHalo selection (Not to use for Cosmic Runs) --- OUTDATED!!!
## process.load('L1TriggerConfig.L1GtConfigProducers.L1GtTriggerMaskTechTrigConfig_cff')
## process.load('HLTrigger/HLTfilters/hltLevel1GTSeed_cfi')

## process.L1T1=process.hltLevel1GTSeed.clone()
## process.L1T1.L1TechTriggerSeeding = cms.bool(True)
## process.L1T1.L1SeedsLogicalExpression = cms.string('(40 OR 41) AND NOT (36 OR 37 OR 38 OR 39)')

#process.TkCalPath = cms.Path(process.L1T1*process.TkCalFullSequence)
process.TkCalPath = cms.Path(process.TkCalFullSequence)


#uncomment to OVERWRITE Gain condition in GT
#process.PoolDBESSource = cms.ESSource("PoolDBESSource",
#    BlobStreamerName = cms.untracked.string('TBufferBlobStreamingService'),
#    DBParameters = cms.PSet(
#        messageLevel = cms.untracked.int32(2),
#        authenticationPath = cms.untracked.string('/afs/cern.ch/cms/DB/conddb')
#    ),
#    timetype = cms.string('runnumber'),
#    toGet = cms.VPSet(cms.PSet(
#        record = cms.string('SiStripApvGainRcd'),
#        tag = cms.string('SiStripGainFromParticles')
#    )),
#    connect = cms.string('sqlite_file:Gains_Sqlite.db')
#)
#process.es_prefer_gainpayload = cms.ESPrefer("PoolDBESSource")




process.options = cms.untracked.PSet(
    Rethrow = cms.untracked.vstring('OtherCMS', 
        'StdException', 
        'Unknown', 
        'BadAlloc', 
        'BadExceptionType', 
        'ProductNotFound', 
        'DictionaryNotFound', 
        'InsertFailure', 
        'Configuration', 
        'LogicError', 
        'UnimplementedFeature', 
        'InvalidReference', 
        'NullPointerError', 
        'NoProductSpecified', 
        'EventTimeout', 
        'EventCorruption', 
        'ScheduleExecutionFailure', 
        'EventProcessorFailure', 
        'FileInPathError', 
        'FileOpenError', 
        'FileReadError', 
        'FatalRootError', 
        'MismatchedInputFiles', 
        'ProductDoesNotSupportViews', 
        'ProductDoesNotSupportPtr', 
        'NotFound')
)
