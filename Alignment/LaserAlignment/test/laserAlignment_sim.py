
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
process.load( "Configuration.StandardSequences.FrontierConditions_GlobalTag_noesprefer_cff" )
#process.load( "Configuration.StandardSequences.FrontierConditions_GlobalTag_cff" )

process.GlobalTag.globaltag = 'IDEAL_V12::All'
#process.GlobalTag.globaltag = 'CRAFT_ALL_V11::All'


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
  connect = cms.string( 'sqlite_file:/afs/cern.ch/user/o/olzem/cms/cmssw/CMSSW_2_2_12/src/Alignment/LaserAlignment/test/Alignments_S.db' )
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
    'file:/afs/cern.ch/user/o/olzem/scratch0/cms/las/prod/nt/TkAlLAS_0.root',
    #'file:/afs/cern.ch/user/o/olzem/scratch0/cms/las/prod/nt/TkAlLAS_1.root',
    #'file:/afs/cern.ch/user/o/olzem/scratch0/cms/las/prod/nt/TkAlLAS_2.root',
    #'file:/afs/cern.ch/user/o/olzem/scratch0/cms/las/prod/nt/TkAlLAS_3.root'
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
    DigiLabel = cms.string( 'VirginRaw' ),
    DigiProducer = cms.string( 'laserAlignmentT0Producer' ), #simSiStripDigis
    DigiType = cms.string( 'Raw' )
  )
)
process.LaserAlignment.SaveToDbase = True
process.LaserAlignment.SaveHistograms = True
process.LaserAlignment.SubtractPedestals = False
process.LaserAlignment.UpdateFromInputGeometry = False
process.LaserAlignment.EnableJudgeZeroFilter = False
process.LaserAlignment.JudgeOverdriveThreshold = 20000
process.LaserAlignment.PeakFinderThreshold = 0.
process.LaserAlignment.ApplyBeamKinkCorrections = False
process.LaserAlignment.MisalignedByRefGeometry = False

## special parameters for LaserAlignment
process.LaserAlignment.ForceFitterToNominalStrips = True


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









