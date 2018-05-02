import FWCore.ParameterSet.Config as cms

process = cms.Process("NOISEGAINBUILDER")
process.MessageLogger = cms.Service("MessageLogger",
    threshold = cms.untracked.string('INFO'),
    destinations = cms.untracked.vstring('cout')
)

process.source = cms.Source("EmptySource",
    numberEventsInRun = cms.untracked.uint32(1),
    firstRun = cms.untracked.uint32(128408)
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)

process.poolDBESSource = cms.ESSource("PoolDBESSource",
   BlobStreamerName = cms.untracked.string('TBufferBlobStreamingService'),
   DBParameters = cms.PSet(
        messageLevel = cms.untracked.int32(2),
        authenticationPath = cms.untracked.string('/afs/cern.ch/cms/DB/conddb')
    ),
    timetype = cms.untracked.string('runnumber'),
    # connect = cms.string('oracle://cms_orcoff_prod/CMS_COND_31X_STRIP'),
    connect = cms.string('sqlite_file:dbfile.db'),
    toGet = cms.VPSet(cms.PSet(
        record = cms.string('SiStripApvGainRcd'),
        # tag = cms.string('SiStripApvGain_GR10_v1_hlt')
        # tag = cms.string('SiStripApvGain_default')
        tag = cms.string('SiStripApvGain_Ideal_31X')
    ))
)

process.PoolDBOutputService = cms.Service("PoolDBOutputService",
    BlobStreamerName = cms.untracked.string('TBufferBlobStreamingService'),
    DBParameters = cms.PSet(
        authenticationPath = cms.untracked.string('/afs/cern.ch/cms/DB/conddb')
    ),
    timetype = cms.untracked.string('runnumber'),
    connect = cms.string('sqlite_file:dbfile.db'),
    toPut = cms.VPSet(cms.PSet(
        record = cms.string('SiStripNoisesRcd'),
        tag = cms.string('SiStripNoiseNormalizedWithIdealGain')
    ))
)

process.SiStripDetInfoFileReader = cms.Service("SiStripDetInfoFileReader")
from SimTracker.SiStripDigitizer.SiStripDigi_cfi import *
process.prod = cms.EDFilter("SiStripNoiseNormalizedWithApvGainBuilder",
                            printDebug = cms.untracked.uint32(5),
                                                                          
                            StripLengthMode = cms.bool(True),

                            #relevant if striplenght mode is chosen
                            # standard value for deconvolution mode is 51. For peak mode 38.8.
                            # standard value for deconvolution mode is 630. For peak mode  414.
                            
                            # TIB
                            NoiseStripLengthSlopeTIB = cms.vdouble( 51.0,  51.0,  51.0,  51.0),
                            NoiseStripLengthQuoteTIB = cms.vdouble(630.0, 630.0, 630.0, 630.0),
                            # TID                         
                            NoiseStripLengthSlopeTID = cms.vdouble( 51.0,  51.0,  51.0),
                            NoiseStripLengthQuoteTID = cms.vdouble(630.0, 630.0, 630.0),
                            # TOB                         
                            NoiseStripLengthSlopeTOB = cms.vdouble( 51.0,  51.0,  51.0,  51.0,  51.0,  51.0),
                            NoiseStripLengthQuoteTOB = cms.vdouble(630.0, 630.0, 630.0, 630.0, 630.0, 630.0),
                            # TEC
                            NoiseStripLengthSlopeTEC = cms.vdouble( 51.0,  51.0,  51.0,  51.0,  51.0,  51.0,  51.0),
                            NoiseStripLengthQuoteTEC = cms.vdouble(630.0, 630.0, 630.0, 630.0, 630.0, 630.0, 630.0),
                            
                            #electronPerAdc = cms.double(1.0),
                            electronPerAdc = simSiStripDigis.electronPerAdc,

                            #relevant if random mode is chosen
                            # TIB
                            MeanNoiseTIB  = cms.vdouble(4.0, 4.0, 4.0, 4.0),
                            SigmaNoiseTIB = cms.vdouble(0.5, 0.5, 0.5, 0.5),
                            # TID
                            MeanNoiseTID  = cms.vdouble(4.0, 4.0, 4.0),
                            SigmaNoiseTID = cms.vdouble(0.5, 0.5, 0.5),
                            # TOB
                            MeanNoiseTOB  = cms.vdouble(4.0, 4.0, 4.0, 4.0, 4.0, 4.0),
                            SigmaNoiseTOB = cms.vdouble(0.5, 0.5, 0.5, 0.5, 0.5, 0.5),
                            # TEC
                            MeanNoiseTEC  = cms.vdouble(4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0),
                            SigmaNoiseTEC = cms.vdouble(0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5),
                            
                            MinPositiveNoise = cms.double(0.1)
)

process.p = cms.Path(process.prod)


