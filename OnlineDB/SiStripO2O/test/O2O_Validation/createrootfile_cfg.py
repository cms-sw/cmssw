import FWCore.ParameterSet.Config as cms

process = cms.Process("Demo")

#process.load("FWCore.MessageService.MessageLogger_cfi")
process.MessageLogger = cms.Service("MessageLogger", 
     destinations = cms.untracked.vstring('O2ORootFileCreation') 
)
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)
process.source = cms.Source("EmptySource",
    numberEventsInRun = cms.untracked.uint32(1),
    firstRun = cms.untracked.uint32(1)
)

process.poolDBESSource = cms.ESSource("PoolDBESSource",
   BlobStreamerName = cms.untracked.string('TBufferBlobStreamingService'),
   DBParameters = cms.PSet(
        messageLevel = cms.untracked.int32(2),
        authenticationPath = cms.untracked.string('/afs/cern.ch/cms/DB/conddb')
    ),
    timetype = cms.untracked.string('runnumber'),
    connect = cms.string('sqlite_file:dbfile.db'),
        toGet = cms.VPSet(
        cms.PSet(
            record = cms.string('SiStripNoisesRcd'),
            tag = cms.string('SiStripNoise_test')
            ),
        cms.PSet(
            record = cms.string('SiStripThresholdRcd'),
            tag = cms.string('SiStripThreshold_test')
          ), 
         cms.PSet(
             record = cms.string('SiStripBadStripRcd'),
             tag = cms.string('SiStripBadChannel_test')
         ),
          cms.PSet(
             record = cms.string('SiStripPedestalsRcd'),
             tag = cms.string('SiStripPedestals_test')
          ),
         cms.PSet(
             record = cms.string('SiStripFedCablingRcd'),
             tag = cms.string('SiStripFedCabling_test')
           ),
         cms.PSet(
             record = cms.string('SiStripApvGainRcd'),
             tag = cms.string('SiStripApvGain_test')
         ),
         cms.PSet(
             record = cms.string('SiStripLatencyRcd'),
             tag = cms.string('SiStripApvLatency_test')
     ))
)

process.load("OnlineDB.SiStripO2O.SiStripO2OValidationParameters_cfi")
process.o2orootfile = cms.EDAnalyzer('SiStripO2OValidationRootFile',
                              process.SiStripO2OValidationParameters
                              )
process.o2orootfile.ValidateNoise = True
process.o2orootfile.ValidateFEDCabling = True
process.o2orootfile.ValidatePedestal = True
process.o2orootfile.ValidateQuality = True
process.o2orootfile.ValidateThreshold = True
process.o2orootfile.ValidateAPVTiming = True
process.o2orootfile.ValidateAPVLatency = True
process.o2orootfile.RootFile ="SiStripO2OValidation.root"

process.p = cms.Path(process.o2orootfile)
