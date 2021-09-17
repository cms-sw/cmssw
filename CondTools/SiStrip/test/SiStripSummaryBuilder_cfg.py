import FWCore.ParameterSet.Config as cms

process = cms.Process("ICALIB")
process.MessageLogger = cms.Service("MessageLogger",
    cerr = cms.untracked.PSet(
        enable = cms.untracked.bool(False)
    ),
    cout = cms.untracked.PSet(
        threshold = cms.untracked.string('INFO')
    ),
    files = cms.untracked.PSet(
        SiStripSummaryBuilder = cms.untracked.PSet(

        )
    )
)

# different !!
process.source = cms.Source("EmptySource",
    timetype = cms.string("runnumber"),
    firstRun = cms.untracked.uint32(1),
    lastRun = cms.untracked.uint32(1),
    interval = cms.untracked.uint32(1)
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)


process.PoolDBOutputService = cms.Service("PoolDBOutputService",
    BlobStreamerName = cms.untracked.string('TBufferBlobStreamingService'),
    DBParameters = cms.PSet(
        authenticationPath = cms.untracked.string('/afs/cern.ch/cms/DB/conddb')
    ),
    timetype = cms.untracked.string('runnumber'),
    connect = cms.string('sqlite_file:dbfile.db'),
    toPut = cms.VPSet(cms.PSet(
        record = cms.string('SiStripSummaryRcd'),#different
        tag = cms.string('SiStripSummary_test1')
    ))
)

#process.siStripPopConHistoricDQM = cms.OutputModule("SiStripPopConHistoricDQM",
#   record = cms.string("SiStripSummaryRcd"),
#   loggingOn = cms.untracked.bool(True),
#   SinceAppendMode = cms.bool(True),
#   Source = cms.PSet(
#      since = cms.untracked.uint32(1),
#      debug = cms.untracked.bool(False))
#) 

process.prod = cms.EDAnalyzer("SiStripSummaryBuilder",
 
   histoList = cms.VPSet(
     
   # quantities are 'stat', 'landau', 'gauss'
   # where 
   #'stat' includes entries, mean, rms
   #'landau' includes
   #'gauss' includes gaussMean, gaussSigma
   
        
   # CKFTk
   #,
   cms.PSet( keyName = cms.untracked.string("Chi2_CKFTk"), quantitiesToExtract = cms.untracked.vstring("stat"))
   ,
   cms.PSet( keyName = cms.untracked.string("NumberOfTracks_CKFTk"), quantitiesToExtract = cms.untracked.vstring("stat"))
   ,
   cms.PSet( keyName = cms.untracked.string("NumberOfRecHitsPerTrack_CKFTk"), quantitiesToExtract = cms.untracked.vstring("stat"))
   
   

   # Summary Cluster Properties
   # ,
   # cms.PSet( keyName = cms.untracked.string("Summary_TotalNumberOfClusters_OffTrack"), quantitiesToExtract = cms.untracked.vstring("stat"))
   # ,
   # cms.PSet( keyName = cms.untracked.string("Summary_TotalNumberOfClusters_OnTrack"),  quantitiesToExtract = cms.untracked.vstring("stat"))
   # ,
   # cms.PSet( keyName = cms.untracked.string("Summary_ClusterChargeCorr_OnTrack"),	quantitiesToExtract = cms.untracked.vstring("stat"))
   # ,
   # cms.PSet( keyName = cms.untracked.string("Summary_ClusterCharge_OffTrack"), 	quantitiesToExtract = cms.untracked.vstring("stat"))
   # ,
   # cms.PSet( keyName = cms.untracked.string("Summary_ClusterNoise_OnTrack"),		quantitiesToExtract = cms.untracked.vstring("stat"))
   # ,
   # cms.PSet( keyName = cms.untracked.string("Summary_ClusterNoise_OffTrack"),  	quantitiesToExtract = cms.untracked.vstring("stat"))
   # ,
   # cms.PSet( keyName = cms.untracked.string("Summary_ClusterStoNCorr_OnTrack"),	quantitiesToExtract = cms.untracked.vstring("stat"))
   # ,
   # cms.PSet( keyName = cms.untracked.string("Summary_ClusterStoN_OffTrack"),		quantitiesToExtract = cms.untracked.vstring("stat"))
   # ,
   # cms.PSet( keyName = cms.untracked.string("Summary_ClusterWidth_OnTrack"),		quantitiesToExtract = cms.untracked.vstring("stat"))
   # ,
   # cms.PSet( keyName = cms.untracked.string("Summary_ClusterWidth_OffTrack"),  	quantitiesToExtract = cms.untracked.vstring("stat"))
      
   ))

process.asciiprint = cms.OutputModule("AsciiOutputModule")

process.p = cms.Path(process.prod)
process.ep = cms.EndPath(process.asciiprint)


