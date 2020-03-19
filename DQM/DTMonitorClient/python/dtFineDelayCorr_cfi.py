import FWCore.ParameterSet.Config as cms

dtFineDelayCorr = cms.EDAnalyzer("DTFineDelayCorr",
    # prescale factor (in luminosity blocks) to perform client analysis
    diagnosticPrescale = cms.untracked.int32(1),
    # run in online environment
    runOnline = cms.untracked.bool(True),
    # kind of trigger data processed by DTLocalTriggerTask
    hwSources = cms.untracked.vstring('TM'),
    # false if DTLocalTriggerTask used LTC digis
    localrun = cms.untracked.bool(True),                         
    # root folder for booking of histograms
    folderRoot = cms.untracked.string(''),
                                 
    # Read old delays from file or from Db
    readOldFromDb = cms.bool(False),
    # Input file name for old delays
    oldDelaysInputFile = cms.string("dtOldFineDelays.txt"),
    # Write new delays to file or to Db
    writeDB = cms.bool(False),
    # output file name
    outputFile = cms.string("dtFineDelaysNew.txt"),
    # Tag for the t0Mean Histograms
    t0MeanHistoTag  = cms.string("TrackCrossingTimeAll"),
    # Hardware Source (TM)
    hwSource = cms.string("TM"),
    # Choose to use Hist Mean or Gaussian Fit Mean
    gaussMean = cms.bool(False),
    # Require Minimum Number Of Entries in the t0Mean Histogram
    minEntries = cms.untracked.int32(300)
                                
    #bxTimeInterval = cms.double(24.95),
    #rangeWithinBX  = cms.bool(True),
    #dbFromTM      = cms.bool(False)
                                    
)


