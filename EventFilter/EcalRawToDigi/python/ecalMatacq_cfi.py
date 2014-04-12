import FWCore.ParameterSet.Config as cms

ecalMatacq = cms.EDProducer("MatacqProducer",
  #List of matacq data file path patterns. %run_number% can be used
  #in place of the run number. 
  #E.g. fileNames = { '/data/matacq_dir1/matacq.%run_number%.dat',
  #                   '/data/matacq_dir2/matacq.%run_number%.dat'}
  #the first pattern than fits is used. In above example if the file
  #is in both matacq_dir1 and matacq_dir2 directories the one in the former
  #directory is used.
  fileNames = cms.vstring(),

  #Instance name to assign to the produced matacq digis
  digiInstanceName = cms.string(''),

  #Instance name to assign to the produced matacq raw data collection
  rawInstanceName = cms.string(''),

  #Swicth for matacq digi production
  produceDigis = cms.bool(True),

  #Switch for matacq raw data collection production
  produceRaw = cms.bool(False),

  #Switch to enable module timing
  timing = cms.untracked.bool(False),

  #debug verbosity level 
  verbosity = cms.untracked.int32(0),

  #Switch to disable any collection production. For test purpose.
  disabled = cms.bool(False),

  #Name of raw data collection the Matacq data must be merge to
  inputRawCollection = cms.InputTag('rawDataCollector'),

  # Switch for merging Matacq raw data with existing raw data
  # within the same collection. If disabled the new collection will 
  # contains only matacq data
  mergeRaw = cms.bool(True),

  # Swicth for disabling trigger type check. When false, matacq data
  # is looked for whatever is the event trigger type. To be use for
  # data with corrupted DCC header detailed trigger type fields.
  ignoreTriggerType = cms.bool(True),

  # Name of output file for the logs.
  logFileName = cms.untracked.string("matacqProducer.log"),

  # Name of log file for execution  timing. If empty, timing logging is disabled. 
  timeLogFile = cms.untracked.string("matacqProducerTime.txt"),


  # Number of event to skip after occurence of an error which is expected
  # to be persitent
  onErrorDisablingEvtCnt = cms.int32(0)
)


