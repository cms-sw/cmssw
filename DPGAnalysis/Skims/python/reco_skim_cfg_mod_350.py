import FWCore.ParameterSet.Config as cms

process = cms.Process('RERECO')

# this is to avoid the postpathendrun probem with same process name (only with http reader)
process.options = cms.untracked.PSet(
    IgnoreCompletely = cms.untracked.vstring('Configuration')
#    TryToContinue = cms.untracked.vstring('Configuration')
)


# for ispy
process.add_(
    cms.Service("ISpyService",
    outputFileName = cms.untracked.string('Ispy.ig'),
    outputMaxEvents = cms.untracked.int32 (1000),
    online = cms.untracked.bool(True),
    debug = cms.untracked.bool(True)
    )
)


# import of standard configurations
process.load('Configuration/StandardSequences/Services_cff')
process.load('FWCore/MessageService/MessageLogger_cfi')
process.load('Configuration/StandardSequences/GeometryIdeal_cff')
process.load('Configuration/StandardSequences/MagneticField_AutoFromDBCurrent_cff')
process.load('Configuration/StandardSequences/RawToDigi_Data_cff')
process.load('Configuration/StandardSequences/Reconstruction_cff')
process.load('DQMOffline/Configuration/DQMOffline_cff')
process.load('Configuration/StandardSequences/EndOfProcess_cff')
process.load('Configuration/StandardSequences/FrontierConditions_GlobalTag_cff')
process.load('Configuration/EventContent/EventContent_cff')
process.load('ISpy/Analyzers/ISpy_Producer_cff')



######### FILTERING Section #############################

# this is for filtering on HLT path
process.hltHighLevel = cms.EDFilter("HLTHighLevel",
     TriggerResultsTag = cms.InputTag("TriggerResults","","HLT"),
#     HLTPaths = cms.vstring('HLT_Activity_L1A'),           # provide list of HLT paths (or patterns) you want
     HLTPaths = cms.vstring('HLT_MinBiasBSC'),           # provide list of HLT paths (or patterns) you want
     eventSetupPathsKey = cms.string(''), # not empty => use read paths from AlCaRecoTriggerBitsRcd via this key
     andOr = cms.bool(True),             # how to deal with multiple triggers: True (OR) accept if ANY is true, False (AND) accept if ALL are true
     throw = cms.bool(True),    # throw exception on unknown path names
     saveTags = cms.bool(False)
)

# this is for filtering based on reco variables
process.skimming = cms.EDFilter("BeamSplash",
    energycuttot = cms.untracked.double(1000.0),
    energycutecal = cms.untracked.double(700.0),
    energycuthcal = cms.untracked.double(700.0),
    ebrechitcollection =   cms.InputTag("ecalRecHit","EcalRecHitsEB"),
    eerechitcollection =   cms.InputTag("ecalRecHit","EcalRecHitsEE"),
    hbherechitcollection =   cms.InputTag("hbhereco"),
    applyfilter = cms.untracked.bool(False)                            
)

# this is for filtering on trigger type

process.load("HLTrigger.special.HLTTriggerTypeFilter_cfi")
# 0=random, 1=physics, 2=calibration, 3=technical
process.hltTriggerTypeFilter.SelectedTriggerType = 1

# this is for filtering on L1 technical trigger bit
process.load('L1TriggerConfig.L1GtConfigProducers.L1GtTriggerMaskTechTrigConfig_cff')
process.load('HLTrigger/HLTfilters/hltLevel1GTSeed_cfi')
process.hltLevel1GTSeed.L1TechTriggerSeeding = cms.bool(True)
process.hltLevel1GTSeed.L1SeedsLogicalExpression = cms.string('32 OR 33 OR 40 OR 41')

#this is for filtering/tagging PhysDecl bit
process.physdecl = cms.EDFilter("PhysDecl",
     applyfilter = cms.untracked.bool(False),
     debugOn = cms.untracked.bool(True)
    )


process.configurationMetadata = cms.untracked.PSet(
    version = cms.untracked.string('$Revision: 1.2 $'),
    annotation = cms.untracked.string('promptReco nevts:1'),
    name = cms.untracked.string('PyReleaseValidation')
)
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(NUMEVENTS)
)
process.options = cms.untracked.PSet(
    Rethrow = cms.untracked.vstring('ProductNotFound'),
    wantSummary = cms.untracked.bool(True)
)

process.source = cms.Source("EventStreamHttpReader",

# streaming##################################################
# in p5 
#   sourceURL = cms.string('http://srv-c2d05-14.cms:22100/urn:xdaq-application:lid=30'),
#   consumerName = cms.untracked.string('DQM Source'),

# tunnel to proxy
# THIS SHOULD BE THE CORRECT FOR OFFLINE ACCESSING THE REVERSE PROXY
#   sourceURL = cms.string('http://cmsdaq0.cern.ch/event-server/urn:xdaq-application:lid=30'),

# special tunnel configuration, need to setup an external tunnel
#   sourceURL = cms.string('http://localhost:22100/urn:xdaq-application:lid=30'),
   sourceURL = SOURCE,
   consumerName = cms.untracked.string('Event Display'),

# direct storage manager
#   sourceURL = cms.string('http://localhost:22100/urn:xdaq-application:service=storagemanager'),
#   consumerName = cms.untracked.string('Event Display'),

# playback###################################################
# in pt5
#    sourceURL = cms.string('http://srv-c2d05-05:50082/urn:xdaq-application:lid=29'),

# tunnel
#   sourceURL = cms.string('http://localhost:50082/urn:xdaq-application:lid=29'),
#   consumerName = cms.untracked.string('Playback Source'),
#################################################################
                            
   consumerPriority = cms.untracked.string('normal'),
   max_event_size = cms.int32(7000000),
   SelectHLTOutput = SELECTHLT,
#   SelectHLTOutput = cms.untracked.string('hltOutputDQM'),
#   SelectHLTOutput = cms.untracked.string('hltOutputExpress'),
   max_queue_depth = cms.int32(5),
   maxEventRequestRate = cms.untracked.double(2.0),
   SelectEvents = cms.untracked.PSet(
#       SelectEvents = cms.vstring('*DQM')
       SelectEvents = cms.vstring('*')
#       SelectEvents = cms.vstring('PhysicsPath')
   ),
   headerRetryInterval = cms.untracked.int32(3)
)



#process.source = cms.Source("PoolSource",
#    debugVerbosity = cms.untracked.uint32(0),
#    debugFlag = cms.untracked.bool(False),
#    fileNames = cms.untracked.vstring(
##'/store/data/Commissioning08/BeamHalo/RECO/StuffAlmostToP5_v1/000/061/642/10A0FE34-A67D-DD11-AD05-000423D94E1C.root'
##
##'/store/express/CRAFT09/ExpressMuon/FEVT/v1/000/110/835/FED0EFCD-AB87-DE11-9B72-000423D99658.root'
##'/store/express/CRAFT09/ExpressMuon/FEVT/v1/000/110/835/FC629BD2-CF87-DE11-9077-001D09F25438.root',
##'/store/express/CRAFT09/ExpressMuon/FEVT/v1/000/110/835/FC38EE75-BD87-DE11-822A-001D09F253C0.root',
##'/store/express/CRAFT09/ExpressMuon/FEVT/v1/000/110/835/FC1CB101-A487-DE11-9F10-000423D99660.root'
#))


process.FEVT = cms.OutputModule("PoolOutputModule",
    maxSize = cms.untracked.int32(1000),
    fileName = cms.untracked.string('EVDISPSM_DIR/EVDISPSM_SUFFIX.root'),
    outputCommands = cms.untracked.vstring('keep *','drop *_MEtoEDMConverter_*_*'),
    dataset = cms.untracked.PSet(
    	      dataTier = cms.untracked.string('RAW-RECO'),
#    	      filterName = cms.untracked.string(''))
    	      filterName = cms.untracked.string('EVDISP')),
                                SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('fullpath')
    )
)

# Other statements
#process.GlobalTag.connect = 'sqlite_file:/afs/cern.ch/user/m/malgeri/public/gtfirstcoll.db'
process.GlobalTag.globaltag = 'GR10_P_V2::All'

process.fifthCkfTrajectoryFilter.filterPset.minimumNumberOfHits = 2
process.fifthCkfTrajectoryFilter.filterPset.maxLostHits = 4
process.fifthCkfTrajectoryFilter.filterPset.maxConsecLostHits = 2
process.fifthCkfInOutTrajectoryFilter.filterPset.minimumNumberOfHits = 2
process.fifthCkfInOutTrajectoryFilter.filterPset.maxLostHits = 4
process.fifthCkfInOutTrajectoryFilter.filterPset.maxConsecLostHits = 2
process.fifthCkfTrajectoryBuilder.minNrOfHitsForRebuild = 2
process.fifthRKTrajectorySmoother.minHits = 2
process.fifthRKTrajectoryFitter.minHits = 2
process.fifthFittingSmootherWithOutlierRejection.MinNumberOfHits = 2
process.tobtecStepLoose.minNumberLayers = 2
process.tobtecStepLoose.maxNumberLostLayers = 2
process.tobtecStepLoose.dz_par1 = cms.vdouble(10.5, 4.0)
process.tobtecStepLoose.dz_par2 = cms.vdouble(10.5, 4.0)
process.tobtecStepLoose.d0_par1 = cms.vdouble(10.5, 4.0)
process.tobtecStepLoose.d0_par2 = cms.vdouble(10.5, 4.0)
process.tobtecStepLoose.chi2n_par = cms.double(100.0)
process.fifthSeeds.RegionFactoryPSet.RegionPSet.originHalfLength = 100
process.fifthSeeds.RegionFactoryPSet.RegionPSet.originRadius     = 10
process.Chi2MeasurementEstimator.MaxChi2 = 100


# to filter on MinBias...
#process.fullpath = cms.Path(process.hltTriggerTypeFilter+process.hltHighLevel+process.RawToDigi+process.reconstruction)

# to filter on trigger type only
#process.fullpath = cms.Path(process.hltTriggerTypeFilter+process.hltHighLevel+process.RawToDigi+process.reconstruction)

#process.fullpath = cms.Path(process.RawToDigi+process.reconstruction+process.skimming+process.iSpy_sequence)
#process.fullpath = cms.Path(process.hltTriggerTypeFilter+process.RawToDigi+process.reconstruction+process.skimming+process.iSpy_sequence)
# added physdecl in tagging mode to catch physdeclared bit in log files
# process.fullpath = cms.Path(process.hltTriggerTypeFilter+process.RawToDigi+process.physdecl+process.reconstruction+process.skimming+process.iSpy_sequence)
process.fullpath = cms.Path(process.RawToDigi+process.physdecl+process.reconstruction+process.skimming+process.iSpy_sequence)

process.out_step = cms.EndPath(process.FEVT)

# Schedule definition

process.schedule = cms.Schedule(process.fullpath,process.out_step)


#process.e = cms.EndPath(process.out)

