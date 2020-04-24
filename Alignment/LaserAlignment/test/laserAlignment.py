
import FWCore.ParameterSet.Config as cms

process = cms.Process( "laserAlignment" )

process.load( "Geometry.CMSCommonData.cmsIdealGeometryXML_cfi" )
process.load( "Geometry.TrackerNumberingBuilder.trackerNumberingGeometry_cfi" )
#process.load( "CondCore.DBCommon.CondDBSetup_cfi" )


## message logger
process.MessageLogger = cms.Service( "MessageLogger",
  debugModules = cms.untracked.vstring( 'LaserAlignment' ),
  cerr = cms.untracked.PSet(
    threshold = cms.untracked.string( 'ERROR' )
  ),
  cout = cms.untracked.PSet(
    threshold = cms.untracked.string( 'INFO' )
  ),
  destinations = cms.untracked.vstring( 'cout', 'cerr' )
)

## all db records

### THIS ONE HAS BEEN LOCALLY MODIFIED!!!
#process.load( "Configuration.StandardSequences.FrontierConditions_GlobalTag_noesprefer_cff" )
#process.load( "Configuration.StandardSequences.FrontierConditions_GlobalTag_cff" )

#process.GlobalTag.globaltag = 'IDEAL_V12::All'
#process.GlobalTag.globaltag = 'CRAFT_ALL_V11::All'


process.load( "Configuration.StandardSequences.FrontierConditions_GlobalTag_cff" )
#process.GlobalTag.globaltag = 'GR09_31X_V5P::All'
process.GlobalTag.globaltag = cms.string('GR_R_37X_V6::All')


## get the tracker alignment records from this file
process.trackerAlignment = cms.ESSource( "PoolDBESSource",
  process.CondDBSetup,
  timetype = cms.string( 'runnumber' ),
  toGet = cms.VPSet(
    cms.PSet(
      record = cms.string( 'TrackerAlignmentRcd' ),
      tag = cms.string( 'Alignments' )
    ), 
    cms.PSet(
      record = cms.string( 'TrackerAlignmentErrorExtendedRcd' ),
      tag = cms.string( 'AlignmentErrorsExtended' )
    )
  ),
  #connect = cms.string( 'sqlite_file:/afs/cern.ch/user/o/olzem/cms/cmssw/CMSSW_2_2_12/src/Alignment/LaserAlignment/test/Alignments_S.db' )
  connect = cms.string( 'sqlite_file:/afs/cern.ch/user/w/wittmer/CMSSW_3_7_0_patch3/src/Alignment/LaserAlignment/test/Alignments.db' )
)

## prefer these alignment record
process.es_prefer_trackerAlignment = cms.ESPrefer( "PoolDBESSource", "trackerAlignment" )

process.load( "Geometry.TrackerGeometryBuilder.trackerGeometry_cfi" )
process.TrackerDigiGeometryESModule.applyAlignment = True


# fast standalone reco output: an sql file
import CondCore.DBCommon.CondDBSetup_cfi
process.PoolDBOutputService = cms.Service( "PoolDBOutputService",
  CondCore.DBCommon.CondDBSetup_cfi.CondDBSetup,
  timetype = cms.untracked.string( 'runnumber' ),
  connect = cms.string( 'sqlite_file:Alignments.db' ),
  toPut = cms.VPSet(
    cms.PSet(
      record = cms.string( 'TrackerAlignmentRcd' ),
      tag = cms.string( 'Alignments' )
    ), 
    cms.PSet(
      record = cms.string( 'TrackerAlignmentErrorExtendedRcd' ),
      tag = cms.string( 'AlignmentErrorsExtended' )
    )
  )
)
process.PoolDBOutputService.DBParameters.messageLevel = 2


## input files
process.source = cms.Source( "PoolSource",
  fileNames = cms.untracked.vstring(
    #'file:/afs/cern.ch/user/o/olzem/scratch0/filterDQM/70664/TkAlLAS.root'
    'file:TkAlLAS_Run140124_LASFilter_test.root'
    #'file:/afs/cern.ch/user/o/olzem/scratch0/cms/las/prod/nt/TkAlLAS_0.root',
    #'file:/afs/cern.ch/user/o/olzem/scratch0/cms/las/prod/nt/TkAlLAS_1.root',
    #'file:/afs/cern.ch/user/o/olzem/scratch0/cms/las/prod/nt/TkAlLAS_2.root',
    #'file:/afs/cern.ch/user/o/olzem/scratch0/cms/las/prod/nt/TkAlLAS_3.root',
    #'file:/afs/cern.ch/user/o/olzem/scratch0/cms/las/prod/nt/TkAlLAS_4.root',
    #'file:/afs/cern.ch/user/o/olzem/scratch0/cms/las/prod/nt/TkAlLAS_5.root',
    #'file:/afs/cern.ch/user/o/olzem/scratch0/cms/las/prod/nt/TkAlLAS_6.root',
    #'file:/afs/cern.ch/user/o/olzem/scratch0/cms/las/prod/nt/TkAlLAS_7.root',
    #'file:/afs/cern.ch/user/o/olzem/scratch0/cms/las/prod/nt/TkAlLAS_8.root',
    #'file:/afs/cern.ch/user/o/olzem/scratch0/cms/las/prod/nt/TkAlLAS_9.root',
    #'file:/afs/cern.ch/user/o/olzem/scratch0/cms/las/prod/nt/TkAlLAS_10.root',
    #'file:/afs/cern.ch/user/o/olzem/scratch0/cms/las/prod/nt/TkAlLAS_11.root',
    #'file:/afs/cern.ch/user/o/olzem/scratch0/cms/las/prod/nt/TkAlLAS_12.root',
    #'file:/afs/cern.ch/user/o/olzem/scratch0/cms/las/prod/nt/TkAlLAS_13.root',
    #'file:/afs/cern.ch/user/o/olzem/scratch0/cms/las/prod/nt/TkAlLAS_14.root',
    #'file:/afs/cern.ch/user/o/olzem/scratch0/cms/las/prod/nt/TkAlLAS_15.root'
  )

)

process.maxEvents = cms.untracked.PSet(
  input = cms.untracked.int32( 2000 )
)

## the LaserAlignment module
process.load( "Alignment.LaserAlignment.LaserAlignment_cfi" )
process.LaserAlignment.DigiProducersList = cms.VPSet(
  cms.PSet(
    DigiLabel = cms.string( 'ZeroSuppressed' ),
    DigiProducer = cms.string( 'laserAlignmentT0Producer' ), #simSiStripDigis
    DigiType = cms.string( 'Processed' )
  )
)
process.LaserAlignment.SaveToDbase = True
process.LaserAlignment.SaveHistograms = True
process.LaserAlignment.SubtractPedestals = False
process.LaserAlignment.UpdateFromInputGeometry = False
process.LaserAlignment.EnableJudgeZeroFilter = True
process.LaserAlignment.JudgeOverdriveThreshold = 200
process.LaserAlignment.PeakFinderThreshold = 2.
process.LaserAlignment.ApplyBeamKinkCorrections = True
process.LaserAlignment.MaskTECModules = (
  # CRAFT run 70664:
  # no-signal modules (dead power groups)
  470405768, 470390664, 470405832, 470390728, 470160520, 470160584,
  # other no-signal modules (low amplitude, etc)
  470045128,
  # TEC+ Ring4 Beam0 (AT shared)
  470307208, 470323592, 470339976, 470356360, 470372744, 470389128, 470405512, 470421896, 470438280,
  # TEC+ Ring4 Beam3 (AT shared)
  470307976, 470324360, 470340744, 470357128, 470373512, 470389896, 470406280, 470422664, 470439048,
  # TEC+ Ring4 Beam5 (AT shared)
  470308488, 470324872, 470341256, 470357640, 470374024, 470390408, 470406792, 470423176, 470439560,
  # TEC- Ring4 Beam0 (AT shared)
  470045064, 470061448, 470077832, 470094216, 470110600, 470126984, 470143368, 470159752, 470176136, 
  # TEC- Ring4 Beam3 (AT shared)
  470045832, 470062216, 470078600, 470094984, 470111368, 470127752, 470144136, 470160520, 470176904,
  # TEC- Ring4 Beam5 (AT shared)
  470046344, 470062728, 470079112, 470095496, 470111880, 470128264, 470144648, 470161032, 470177416
)
process.LaserAlignment.MaskATModules = (
  # CRAFT run 70664:
  # no-signal modules (too high/low amplitude, etc)
  470373004, 470110852, 470094732, 470111116, 470095236, 470111620, 470112132, 470112396, 
  # TEC(AT)+ Beam0 (TEC shared)
  470307208, 470323592, 470339976, 470356360, 470372744,
  # TEC(AT)+ Beam3 (TEC shared)
  470307976, 470324360, 470340744, 470357128, 470373512,
  # TEC(AT)+ Beam5 (TEC shared)
  470308488, 470324872, 470341256, 470357640, 470374024,
  # TEC(AT)- Beam0 (TEC shared)
  470045064, 470061448, 470077832, 470094216, 470110600,
  # TEC(AT)- Beam3 (TEC shared)
  470045832, 470062216, 470078600, 470094984, 470111368,
  # TEC(AT)- Beam5 (TEC shared)
  470046344, 470062728, 470079112, 470095496, 470111880
)


## special parameters for LaserAlignment
process.LaserAlignment.ForceFitterToNominalStrips = False


## the output file containing the TkLasBeamCollection
## for the track based interface
process.out = cms.OutputModule( "PoolOutputModule",
  fileName = cms.untracked.string( 'tkLasBeams.root' ),
  outputCommands = cms.untracked.vstring(
    'drop *',
    "keep TkLasBeams_*_*_*"
  )
)




## for debugging
process.dump = cms.EDAnalyzer("EventContentAnalyzer")

process.alignment = cms.Sequence( process.LaserAlignment )
process.laser = cms.Path( process.alignment )
process.e = cms.EndPath( process.out )









